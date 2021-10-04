# ResMARL
Experimenting with different MARL algorithms trying to evaluate their resilience level.

Mainly building on top of the SSD domains repo: https://github.com/eugenevinitsky/sequential_social_dilemma_games

Specific additions, we are experimenting with:
1. Agents that can communicate over the environment.
2. Environments that are randomly perturbed over time.
3. Agents that try to minimize global and self confusion levels via messaging.

simply running train.py will run with 2 communicating agents on the cleanup environment.
check default_args.py for changing the running parameters for train.py.
