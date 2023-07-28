import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from onpolicy.algorithms.utils.util import check, init
from onpolicy.algorithms.utils.transformer_act import discrete_autoregreesive_act
from onpolicy.algorithms.utils.transformer_act import discrete_parallel_act
from onpolicy.algorithms.utils.transformer_act import continuous_autoregreesive_act
from onpolicy.algorithms.utils.transformer_act import continuous_parallel_act

from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.gnn import GNNBase

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        # print("self.query: ", self.query)
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):

        print(" B L D : ", query.size())    # batch size, sequence length, dimension
        if len(query.size()) == 2:
           query = query.unsqueeze(1)
            
        print("B L D : ", query.size())    # batch size, sequence length, dimension
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.squeeze(1) # remove first channel 
        # print('att output --', att.size())
        
        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        print('encode block x size: ', x.size())
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x)) # x is [B, num_agents, D]	
        
        # x = x.unsqueeze(1)
        # x = x.expand(-1, rep_enc.size(1), -1, -1) # becomes [B, L, num_agents, D] 
        
        ''' potential work around: create padding and mask before passing to attn2? or force sequence length = num_agents? '''
        # size_difference = rep_enc.size(1) - x.size(1)
        # x = torch.nn.functional.pad(x, (0, 0, 0, size_difference))
        # print('x size: ', x.size())
        
        # bug: x is [B, num_agents, D] and rep_enc is [B, L, D] so the dimensions don't match
        
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc)) # looks like masking is set to True in initialization	
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state): 

        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs): # second channel is actually 12 (not n_agent)
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        print('state', state.shape)
        print('obs', obs.shape) 
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            print("state_embeddings shape: ", state_embeddings.shape)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            print("obs_embeddings shape: ", obs_embeddings.shape)
            x = obs_embeddings
        
        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)


        return logit


class MultiAgentTransformer(nn.Module):

    # def __init__(self, args,
    #                         obs_space,
    #                         node_obs_space,
    #                         edge_obs_space, 
    #                         act_space,
    #                         split_batch, 
    #                        max_batch_size,state_dim, obs_dim, action_dim, n_agent,
    #              n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
    #              action_type='Discrete', dec_actor=False, share_actor=False):
    def __init__(self, args,
                        obs_space,
                        node_obs_space,
                        edge_obs_space, 
                        act_space,
                        split_batch, 
                        max_batch_size,
                        shared_obs_dim, obs_dim, action_dim, n_agent,
                        n_block, n_embd, n_head, 
                        encode_state=False, device=torch.device("cpu"),
                        action_type='Discrete', dec_actor=False, 
                        share_actor=False):
        
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
     
        # state unused
        state_dim = 37

        obs_shape = get_shape_from_obs_space(obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[1] # returns (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]       # returns (edge_dim,)
        
        # print parmaeters shape before moving to GNNBase
        print(" parameters shape before moving to GNNBase:")
        print("obs_shape: ", obs_shape)
        print("node_obs_shape: ", node_obs_shape)
        print("edge_dim: ", edge_dim)
        
        # Look good to this point 
        self.graph_base = GNNBase(args, node_obs_shape, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.graph_base.out_dim # output shape from gnns
        state_dim = gnn_out_dim + state_dim

        print(" here the obs dim is this is the one that mlp layer in the inforMARL had to over-ride", obs_dim)
        
        print(" I am sending the following to encoder: ")
        print("state_dim: ", state_dim)
        print("obs_dim: ", obs_dim)
        print("n_block: ", n_block)
        print("n_embd: ", n_embd)
        print("n_head: ", n_head)
        print("n_agent: ", n_agent)
        print("encode_state: ", encode_state)
        
        
        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        
 
        self.to(device)

        
        
    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    # def forward(self, state, obs, action, available_actions=None):
    def forward(self, obs, node_obs, adj, agent_id, state, action, available_actions=None):


        # state unused
        ori_shape = np.shape(state)
        
        
        print(" The Orginal State Shape is: ", ori_shape)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)
        
        adj = check(adj).to(**self.tpdv)
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        
        node_obs = check(node_obs).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
    
        graph_features = self.graph_base(node_obs, adj, agent_id)
        obs_rep = torch.cat([obs, graph_features], dim=1)
        
        
        batch_size = np.shape(state)[0]
        v_loc, obs_rep = self.encoder(state, obs_rep) # probably here where we would want to incorporate heterogeneity? would would want to torch.concat output 
    
        print("v_loc shape: ", v_loc.shape)
        print("obs_rep shape: ", obs_rep.shape)
        
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        
        print("Inside the ma transformer State Shape before: ", state.shape)
        print("Inside the ma transformer Obs Shape before: ", obs.shape)
        
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 22), dtype=np.float32)

        print()
        print("****************************************************************************************")         
        print("we want to see the state shape", state.shape)   
        print("************************* ************************************ *************************")
        print()   
        


        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        
        
        print("State Shape: ", state.shape)
        print("Obs Shape: ", obs.shape)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)
            
        print("output_action shape: ", output_action.shape)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        # ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        # state = check(state).to(**self.tpdv)
        # obs = check(obs).to(**self.tpdv)
        print(" I am ma transformer and I am getting values")
    
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot



