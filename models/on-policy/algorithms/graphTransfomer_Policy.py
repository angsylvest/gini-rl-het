import gym
import argparse

import torch
from torch import Tensor
from typing import Tuple
from onpolicy.utils.util import update_linear_schedule

from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space

from onpolicy.algorithms.ma_transformer import MultiAgentTransformer as MAT
from onpolicy.algorithms.utils.util import check
import numpy as np
class GR_MAPPOPolicy:
    """
        MAPPO Policy  class. Wraps actor and critic networks 
        to compute actions and value function predictions.

        args: (argparse.Namespace) 
            Arguments containing relevant model and policy information.
        obs_space: (gym.Space) 
            Observation space.
        cent_obs_space: (gym.Space) 
            Value function input space 
            (centralized input for MAPPO, decentralized for IPPO).
        node_obs_space: (gym.Space)
            Node observation space
        edge_obs_space: (gym.Space)
            Edge dimension in graphs
        action_space: (gym.Space) a
            Action space.
        device: (torch.device) 
            Specifies the device to run on (cpu/gpu).
    """

    def __init__(self, 
                args:argparse.Namespace, 
                obs_space:gym.Space,
                cent_obs_space:gym.Space,
                node_obs_space:gym.Space,
                edge_obs_space:gym.Space,
                act_space:gym.Space,
                device=torch.device("cpu")) -> None:
        
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.node_obs_space = node_obs_space
        self.edge_obs_space = edge_obs_space
        self.act_space = act_space
     
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.split_batch = args.split_batch
        self.max_batch_size = args.max_batch_size
        self.num_agents = args.num_agents
        
        # getting the name of the action space
        if act_space.__class__.__name__ == 'Box':
                self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'    
            
        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            print("act dim in the graphTransfomer_policy file: ", self.act_dim)
            self.act_num = self.act_dim    

        print("obs_space: ", obs_space.shape)

            
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]   
            
        # we want to use the transfomer stuff here with the graph neighbors information 
       
       
        print("parameters for the transformer that we are passing to MAT:")
        print("obs_space:", self.obs_space.shape)
        print("node_obs_space:", self.node_obs_space.shape)
        print("edge_obs_space:", self.edge_obs_space.shape)
        print("act_space:", self.act_space.shape)
        
         
        self.transformer = MAT(args,
                            obs_space,
                            node_obs_space,
                            edge_obs_space, 
                            act_space,
                            
                            self.split_batch, 
                            self.max_batch_size,
                            
                            self.share_obs_dim, self.obs_dim, self.act_dim, args.num_agents,
                            n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                            encode_state=args.encode_state, device=device,
                            action_type=self.action_type, dec_actor=args.dec_actor,
                            share_actor=args.share_actor)
            
            
        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)        
        
        


    def lr_decay(self, episode:int, episodes:int) -> None:
        """
            Decay the actor and critic learning rates.
            episode: (int) 
                Current training episode.
            episodes: (int) 
                Total number of training episodes.
        """
        update_linear_schedule(optimizer=self.actor_optimizer, 
                            epoch=episode, 
                            total_num_epochs=episodes, 
                            initial_lr=self.lr)
        update_linear_schedule(optimizer=self.critic_optimizer, 
                            epoch=episode, 
                            total_num_epochs=episodes, 
                            initial_lr=self.critic_lr)

    def get_actions(self, 
                    cent_obs, 
                    obs,
                    node_obs,
                    adj,
                    agent_id,
                    share_agent_id,
                    rnn_states_actor, 
                    rnn_states_critic, 
                    masks, 
                    available_actions=None,
                    deterministic=False) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        
        # This resloved the issue with    assert len(data) == 2 and AttributeError: 'numpy.ndarray' object has no attribute 'dim'
        
        
        print()
        print("****************************************************************************************")
        print("In the get action from the policy graph transformer, we are getting the following")
        print(" the original cent obs shape: ", cent_obs.shape)
        print(" the original obs shape: ", obs.shape)
        print(" the original node obs shape: ", node_obs.shape)
        print(" the original adj shape: ", adj.shape)
        print("************************* SO FAR SO GOOD (similar to informarl) *************************")
        print()
        
        adj = check(adj).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)


        print('actor critic shapes', rnn_states_actor.shape, rnn_states_critic.shape)
        
        
        nbd_features = self.transformer.graph_base(node_obs, adj, agent_id)
        rnn_states_actor = rnn_states_actor.squeeze(1)
        rnn_states_critic = rnn_states_critic.squeeze(1)
        
        #actor_features = torch.cat([cent_obs, nbd_features, rnn_states_actor], dim=1)
        actor_features = torch.cat([obs, nbd_features], dim=1)
        
        print()
        print("****************************************************************************************")         
        print(" Here is what we have after concatination: actor_features shape: ", actor_features.shape) # [384, 22]
        print("************************* ************************************ *************************")
        print()   
        
        
        critic_features = nbd_features
        
        # padded zeros to make the shape a multiple of 6 for the transformer
        device = obs.device
        zeros_tensor = torch.zeros(obs.shape[0], 2, device=device)
        critic_features = torch.cat([critic_features, zeros_tensor], dim=1)
        
        # critic_features = torch.cat([obs, rnn_states_critic, zeros_tensor], dim=1)

                
        
        
        print()
        print("****************************************************************************************")         
        print(" Here is what we have critic feature: ", critic_features.shape)   #  Instead of this [384, 72] we go for this [384, 18]
        print("************************* ************************************ *************************")
        print()   
        
        # actor_features = actor_features.unsqueeze(1)
        # critic_features = critic_features.view(critic_features.shape[0], -1, 6)
        
        actor_features = actor_features.reshape(-1, self.num_agents, actor_features.shape[1]) # [384, 3, 22]
        # print('self.obs_dim:', self.obs_dim)
        
        
        # [384, 3, 6] I don't know why we are forced to make it self.obs_dim = 6 instead of self.critic_feature[1] = 18
        critic_features = critic_features.reshape(-1, self.num_agents, self.obs_dim) 
        print('After actor_features:', actor_features.shape)
        print('After critic_features:', critic_features.shape)
        
        
        # # this is the average pooling layer
        # import torch.nn as nn
        # pool = nn.AvgPool1d(kernel_size=4, stride=4)
        # critic_features = pool(critic_features.transpose(1, 2)).transpose(1, 2)
       
        actions, action_log_probs, values = self.transformer.get_actions(actor_features,
                                                                         critic_features, #obs is replaced by actor_features
                                                                         available_actions,
                                                                         deterministic)
       
       
        print(" values after calling self.transformer.get_actions():", values.shape)
       
        # transform the actions is giving me double the action that I would get for the gnn 
        print('actions output from self.transformer.get_actions', actions.shape)
        
        
        print('actions output from the graphTransformer_Policy.py:', actions.shape)
        # this prints out actions output from the graphTransformer_Policy.py: torch.Size([384, 3, 1])
        
        actions = actions.view(-1, self.act_num)  # action = action.view(384, -1)
        # here the output will be  384x3 = 1151 torch.Size([1152, 1])
        
        
        
        print(" after doing actions = actions.view(-1, self.act_num)", actions.shape)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        # print(" ************ after doing values = values.view(-1, 1)", values.shape)
        # unused, just for compatibility
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        
        print()
        print(" rnn_states_actor:", rnn_states_actor.shape)
        print(" rnn_states_critic:", rnn_states_critic.shape)
        
        self.rnn_states_actor = rnn_states_actor
        self.rnn_states_critic = rnn_states_critic
        print(" rnn_ state shape this is the one with problem", rnn_states_actor.shape)
        print("get action DONE!!!")
        
        
        # this two resloved the error ValueError: shape mismatch: 
        # value array of shape (0,1,64) could not be broadcast to indexing result of shape (0,64)
        
        rnn_states_actor = rnn_states_actor.unsqueeze(1)
        rnn_states_critic = rnn_states_critic.unsqueeze(1)
        print(" rnn_ state shape this is the after the problem", rnn_states_actor.shape)              
        return (values, actions, action_log_probs, 
                rnn_states_actor, rnn_states_critic)

    def get_values(self, 
                    cent_obs,
                    node_obs,
                    adj,
                    share_agent_id,
                    rnn_states_critic, 
                    masks) -> Tensor:
        """
            Get value function predictions.
            cent_obs (np.ndarray): 
                centralized input to the critic.
            node_obs (np.ndarray): 
                Local agent graph node features to the actor.
            adj (np.ndarray): 
                Adjacency matrix for the graph.
            share_agent_id (np.ndarray): 
                Agent id to which cent_observations belong to.
            rnn_states_critic: (np.ndarray) 
                if critic is RNN, RNN states for critic.
            masks: (np.ndarray) 
                denotes points at which RNN states should be reset.

            :return values: (torch.Tensor) value function predictions.
        """
        #agent_id is changed to share_agent_id
        
        # Ensure all inputs are PyTorch tensors and are on the correct device
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        share_agent_id = check(share_agent_id).to(**self.tpdv).long()
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)

        rnn_states_critic = rnn_states_critic.squeeze(1)
        
        # Extract neighbourhood features torch.Size([384, 48])
        nbd_features = self.transformer.graph_base(node_obs, adj, share_agent_id)
        
        
        print('nbd_features shape in value function:', nbd_features.shape)
        
        
        # Create actor_features torch.Size([384, 130])
        actor_features = torch.cat([cent_obs, nbd_features, rnn_states_critic], dim=1)
        
        # Check actor_features shape 
        print('actor_features shape before unsqueeze:', actor_features.shape)
        actor_features = actor_features.unsqueeze(1)
        
        # torch.Size([384, 1, 130])
        print('actor_features shape:', actor_features.shape)
        
        # Create a tensor of zeros of shape [actor_features.shape[0], 1, 2] and concatenate it with actor_features
        zero_padding = torch.zeros(actor_features.shape[0], 1, 2).to(**self.tpdv)
        actor_features_padded = torch.cat([actor_features, zero_padding], dim=2)

        # actor_features_padded shape: torch.Size([384, 1, 132])
        print('actor_features_padded shape:', actor_features_padded.shape)

        # Reshape the actor_features tensor to have its last dimension of size 6
        actor_features= actor_features_padded.view(actor_features_padded.shape[0], -1, 6)

        # actor_features shape after view: torch.Size([384, 22, 6])   <-    22?? BIG PROBLEM HERE, I NEED TO FIX THIS
        print('actor_features shape after view:', actor_features.shape)
        
        ori_shape = cent_obs.shape
        state = torch.zeros((*ori_shape[:-1], 37), dtype=torch.float32)  # The last dimension might need to be adjusted
        state = check(state).to(**self.tpdv)
            
        # obs = check(obs).to(**self.tpdv)
        print('updated features shape', actor_features.shape)
        
        v_tot = self.transformer.get_values(state, actor_features) #2nd argument was obs changed to actor_features
        print('v_tot shape:', v_tot.shape)
        
        # --- downsample sequence length to 1 in L (channel 2 dim) ---
        import torch.nn as nn
        v_tot = v_tot.permute(0, 2, 1)
        pool = nn.AvgPool1d(22) 
        
        v_tot = pool(v_tot).squeeze(1)
        print('v_tot shape: updated', v_tot.shape)
        
        # ---------------------------------------------------------------
        
        return v_tot
    


    def evaluate_actions(self, 
                        cent_obs, 
                        obs,
                        node_obs,
                        adj,
                        agent_id,
                        share_agent_id,
                        rnn_states_actor, 
                        rnn_states_critic, 
                        action = None, 
                        masks = None,
                        available_actions=None, 
                        active_masks=None) -> Tuple[Tensor, Tensor, Tensor]:
        """
            Get action logprobs / entropy and 
            value function predictions for actor update.
            cent_obs (np.ndarray): 
                centralized input to the critic.
            obs (np.ndarray): 
                local agent inputs to the actor.
            node_obs (np.ndarray): 
                Local agent graph node features to the actor.
            adj (np.ndarray): 
                Adjacency matrix for the graph.
            agent_id (np.ndarray): 
                Agent id for observations
            share_agent_id (np.ndarray): 
                Agent id for shared observations
            rnn_states_actor: (np.ndarray) 
                if actor is RNN, RNN states for actor.
            rnn_states_critic: (np.ndarray) 
                if critic is RNN, RNN states for critic.
            action: (np.ndarray) 
                actions whose log probabilites and entropy to compute.
            masks: (np.ndarray) 
                denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) 
                denotes which actions are available to agent
                (if None, all actions available)
            active_masks: (torch.Tensor) 
                denotes whether an agent is active or dead.

            :return values: (torch.Tensor) 
                value function predictions.
            :return action_log_probs: (torch.Tensor) 
                log probabilities of the input actions.
            :return dist_entropy: (torch.Tensor) 
                action distribution entropy for the given inputs.
        """
        

        print('nbd features in evaluate actions ', node_obs.shape, adj.shape, share_agent_id.shape)
        nbd_features = torch.Tensor(self.transformer.graph_base(node_obs, adj, share_agent_id))
        actor_features = torch.cat([torch.Tensor(obs), nbd_features], dim=1)
        
        print('cent obs shape in evaluate actions', cent_obs.shape)
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        
        actor_features = actor_features.reshape(-1, self.num_agents, self.obs_dim)
        
        actions = actions.reshape(-1, self.num_agents, self.act_num)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        action_log_probs, values, entropy = self.transformer(cent_obs, obs, actions, available_actions)

        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        print('evalute actions output shape', values.shape, action_log_probs.shape, entropy.shape)
        return values, action_log_probs, entropy
    


    def act(self, 
            obs,
            node_obs,
            adj,
            agent_id,
            rnn_states_actor, 
            masks, 
            available_actions=None, 
            deterministic=False) -> Tuple[Tensor, Tensor]:
        """
            Compute actions using the given inputs.
            obs (np.ndarray): 
                local agent inputs to the actor.
            node_obs (np.ndarray): 
                Local agent graph node features to the actor.
            adj (np.ndarray): 
                Adjacency matrix for the graph.
            agent_id (np.ndarray): 
                Agent id for nodes for the graph.
            rnn_states_actor: (np.ndarray) 
                if actor is RNN, RNN states for actor.
            masks: (np.ndarray) 
                denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) 
                denotes which actions are available to agent
                (if None, all actions available)
            deterministic: (bool) 
                whether the action should be mode of 
                distribution or should be sampled.
        """
        # adj has to be changed to torch tensor everywher in this file where we have called graph_base
        nbd_features = self.transformer.graph_base(node_obs, adj, agent_id)
        actor_features = torch.cat([obs, nbd_features], dim=1)
        rnn_states_critic = self.rnn_states_critic
        rnn_states_actor = self.rnn_states_actor
        _, actions, _, rnn_states_actor, _ = self.get_actions(obs,
                                                              actor_features,
                                                              rnn_states_actor,
                                                              rnn_states_critic,
                                                              masks,
                                                              available_actions,
                                                              deterministic)
        print('got actions ----', actions.shape)
        return actions, rnn_states_actor

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()