# Gini Lab 2023: Basic skeleton of policy 

# import relevant libraries 
import logging
from abc import ABC
from typing import Dict
from typing import List, Optional, Union
from typing import Type

import gym
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.execution import synchronous_parallel_sample
from ray.rllib.execution.common import (
    _check_sample_batch_type,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.models import ModelV2, ActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
    SampleBatch,
    DEFAULT_POLICY_ID,
    concat_samples,
)
from ray.rllib.policy.torch_mixins import (
    LearningRateSchedule,
    KLCoeffMixin,
    EntropyCoeffSchedule,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
)
from ray.rllib.utils.torch_utils import (
    warn_if_infinite_kl_divergence,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import AgentID, TensorType, ResultDict
from ray.rllib.utils.typing import PolicyID, SampleBatchType

torch, nn = try_import_torch()


class HeteroPolicy(PPOTorchPolicy, MultiAgentValueNetworkMixIn): 
	def __init__(self, observation_space, action_space, config): 
		config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
		super().__init__(observation_space, action_space, config)

	def compute_actions(self, obs_batch, state_batches, prev_act_batch = None, prev_reward_batch = None, info_batch = None, **kwargs): 
		actions = {}
		for agt_id, obs in obs_batch.items():
			action = self.__select_action(agent_id, obs)
			actions[agent_id] = action 
		return actions, state_batches 

	def __select_action(self, agent_id, obs): 
		# decide on action depending on observation 
		return np.random.choice(self.action_space.n)
