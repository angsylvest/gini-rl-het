import os
import pickle

from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import MultiCallbacks

config = {
	"env": "custom_env", 
	"env_config": env_config, 
	"model": {
		"custom_model": my_model
	} 
} 

trainer = PPOTrainer(config = config)

TrainingUtils.init_ray(scenario_name=scenario_name)

tune.run(trainer, callbacks = [])


