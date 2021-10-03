from social_dilemmas.envs.cleanup import CleanupEnv, CleanupEnvWithMessages
from social_dilemmas.envs.harvest import HarvestEnv, HarvestEnvWithMessages
from social_dilemmas.envs.switch import SwitchEnv


def get_env_creator(env, num_agents, args):
    # TODO add here envs with messages and confusion classes
    if env == "harvest":

        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
            )

    elif env == "cleanup":

        def env_creator(_):
            return CleanupEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
            )

    if env == "harvest_msg":

        def env_creator(_):
            return HarvestEnvWithMessages(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
            )

    elif env == "cleanup_msg":

        def env_creator(_):
            return CleanupEnvWithMessages(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
            )

    elif env == "switch":

        def env_creator(_):
            return SwitchEnv(num_agents=num_agents, args=args)

    return env_creator
