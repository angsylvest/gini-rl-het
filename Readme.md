# Onpolicy Algorithms for Multi-agent Reinforcement Learning

This repository contains the implementation of on-policy algorithms for multi-agent reinforcement learning.

## Prerequisites

You need to have `conda` and `pip` installed in your system. 

## Setting Up

Follow the instructions below to set up the project:

1. Create a new conda environment:
   ```shell
   conda create --name myenv python
    ```
- Don't forget to replace myenv with the name you want to give to your environment. 

2. Activate the environment:
   ```shell
   conda activate myenv
    ```
3. Install the required packages:
   ```shell
   pip install -r requirements.txt
    ```
4. Install the package:
   ```shell
    pip install -e .
    ```
## Running the Code

Go to the `train` directory and run the following command to train the agent:

For MPE Comm:
```shell
bash train_mpe_comm.sh
```
For MPE Reference:
```shell
bash train_mpe_reference.sh
```
For MPE Spread:
```shell
bash train_mpe_spread.sh
```

## Project Structure

The project has the following structure:

- onpolicy/
    - algorithms/
        - __init__.py
        - r_mappo/
            - __init__.py
            - algorithm/
                - r_actor_critic.py
                - rMAPPOPolicy.py
            - r_mappo.py
        - runner/
            - separated/
                - base_runner.py
                - mpe_runner.py
            - shared/
                - base_runner.py
                - mpe_runner.py
        - utils/
            - act.py
            - cnn.py
            - distributions.py
            - mlp.py
            - popart.py
            - rnn.py
            - util.py
    - envs/
        - __init__.py
        - env_wrappers.py
        - mpe/
            - __init__.py
            - core.py
            - environment.py
            - MPE_env.py
            - multi_discrete.py
            - rendering.py
            - scenario.py
            - scenarios/
                - __init__.py
                - simple_adversary.py
                - simple_attack.py
                - simple_crypto.py
                - simple_crypto_display.py
                - simple_push.py
                - simple_reference.py
                - simple_speaker_listener.py
                - simple_spread.py
                - simple_tag.py
                - simple_world_comm.py
    - train/
        - __init__.py
        - config.py
        - render/
            - render_mpe.py
            - render_mpe.sh
        - results/
        - train_mpe_comm.sh
        - train_mpe.py
        - train_mpe_reference.sh
        - train_mpe_spread.sh
    - utils/
        - __init__.py
        - multi_discrete.py
        - separated_buffer.py
        - shared_buffer.py
        - util.py
        - valuenorm.py
    - __init__.py
    - Readme.md
    - requirements.txt
    - setup.py


















<!-- 
onpolicy/
│
├── algorithms/
│   ├── __init__.py
│   ├── r_mappo/
│   │   ├── __init__.py
│   │   ├── algorithm/
│   │   │   ├── r_actor_critic.py
│   │   │   └── rMAPPOPolicy.py
│   │   └── r_mappo.py
│   │   
│   ├── runner/
│   │   ├── separated/
│   │   │   ├── base_runner.py
│   │   │   └── mpe_runner.py
│   │   └── shared/
│   │       ├── base_runner.py
│   │       └── mpe_runner.py
│   │   
│   └── utils/
│       ├── act.py
│       ├── cnn.py
│       ├── distributions.py
│       ├── mlp.py
│       ├── popart.py
│       ├── rnn.py
│       └── util.py
│
├── envs/
│   ├── __init__.py
│   ├── env_wrappers.py
│   └── mpe/
│       ├── __init__.py
│       ├── core.py
│       ├── environment.py
│       ├── MPE_env.py
│       ├── multi_discrete.py
│       ├── rendering.py
│       ├── scenario.py
│       └── scenarios/
│           ├── __init__.py
│           ├── simple_adversary.py
│           ├── simple_attack.py
│           ├── simple_crypto.py
│           ├── simple_crypto_display.py
│           ├── simple_push.py
│           ├── simple_reference.py
│           ├── simple_speaker_listener.py
│           ├── simple_spread.py
│           ├── simple_tag.py
│           └── simple_world_comm.py
│
├── train/
│   ├── __init__.py
│   ├── config.py
│   ├── render/
│   │   ├── render_mpe.py
│   │   └── render_mpe.sh
│   ├── results/
│   ├── train_mpe_comm.sh
│   ├── train_mpe.py
│   ├── train_mpe_reference.sh
│   └── train_mpe_spread.sh
│
├── utils/
│   ├── __init__.py
│   ├── multi_discrete.py
│   ├── separated_buffer.py
│   ├── shared_buffer.py
│   ├── util.py
│   └── valuenorm.py
│
├── __init__.py
├── Readme.md
├── requirements.txt
└── setup.py -->
