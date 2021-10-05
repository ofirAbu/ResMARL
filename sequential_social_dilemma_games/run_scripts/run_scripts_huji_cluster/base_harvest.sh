#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c8
#SBATCH --time=72:0:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o base_harvest_log.out
#SBATCH --gres=gpu:2,vmem:10g
#SBATCH --killable
#SBATCH --requeue

workspace="/cs/labs/jeff/ofir.abu"

cd $workspace/ResMARL/sequential_social_dilemma_games/run_scripts/
python train.py \
--env harvest \
--model baseline \
--algorithm PPO \
--num_agents 4 \
--num_workers 6 \
--rollout_fragment_length 1000 \
--num_envs_per_worker 16 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((160 * 10 ** 9)) \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--gpus_for_driver 2 \
--cpus_for_driver 0 \
--num_samples 5 \
--entropy_coeff 0.000687 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights .00136 .000028
deactivate