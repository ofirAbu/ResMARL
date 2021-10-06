#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c12
#SBATCH --time=72:0:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o global_conf_cleanup_50_log.out
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --killable
#SBATCH --requeue

workspace="/cs/labs/jeff/ofir.abu"
module load cuda/10.1
module load cudnn

source $workspace/venvs/res_marl_3.7/bin/activate

cd $workspace/ResMARL/sequential_social_dilemma_games/run_scripts

python train.py \
--env cleanup_pert_50_msg_global \
--model global_confusion \
--algorithm A3C \
--num_agents 4 \
--num_workers 3 \
--rollout_fragment_length 1000 \
--num_envs_per_worker 5 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((160 * 10 ** 9)) \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--gpus_for_driver 1 \
--cpus_for_driver 0 \
--num_samples 3 \
--entropy_coeff 0.00223 \
--moa_loss_weight 0.091650628 \
--messages_loss_weight 0.091650628 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights 0.0012 0.000044 \
--influence_reward_weight 2.521 \
--influence_reward_schedule_steps 0 10000000 100000000 300000000 \
--messages_reward_schedule_steps 0 10000000 100000000 300000000 \
--influence_reward_schedule_weights 0.0 0.0 1.0 0.5 \
--messages_reward_schedule_weights 0.0 0.0 1.0 0.5
deactivate