#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c12
#SBATCH --time=72:0:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o base_cleanup_log.out
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --killable
#SBATCH --requeue

workspace="/cs/labs/jeff/ofir.abu"
module load cuda/10.1
module load cudnn

source $workspace/venvs/res_marl_3.7/bin/activate

cd $workspace/ResMARL/sequential_social_dilemma_games/run_scripts

python train.py \
--env cleanup \
--model baseline \
--algorithm PPO \
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
--entropy_coeff 0.00176 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights .00126 .000012
deactivate