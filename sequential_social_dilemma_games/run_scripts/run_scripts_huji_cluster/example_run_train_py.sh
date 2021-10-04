#!/bin/csh
module load cuda
module load cudnn
source /cs/labs/jeff/ofir.abu/venvs/res_marl_3.7/bin/activate.csh

cd /cs/labs/jeff/ofir.abu/ResMARL/sequential_social_dilemma_games/run_scripts
python train.py
deactivate