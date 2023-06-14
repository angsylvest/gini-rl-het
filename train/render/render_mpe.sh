

#!/bin/sh


# # ************************** this one is the infrence for train_comm ***********************************#
# env="MPE"
# scenario="simple_speaker_listener"
# num_landmarks=3
# num_agents=2
# algo="rmappo"
# exp="check"
# seed_max=1

# echo "env is ${env}"
# pwd
# for seed in `seq ${seed_max}`
# do
#     CUDA_VISIBLE_DEVICES=0 python render_mpe.py --save_gifs --share_policy --env_name ${env} --algorithm_name ${algo} \
#     --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
#     --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 \
#     --model_dir "../scripts/results/MPE/simple_speaker_listener/rmappo/check/wandb/latest-run/files" 
# done




# ************************** this one is the infrence for train_comm ***********************************#
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
pwd
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render_mpe.py --save_gifs  --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 --wandb_name "gini-lab" --user_name "ebagirma8"\
    --model_dir "../results/MPE/simple_spread/rmappo/check/wandb/latest-run/files" 
done


