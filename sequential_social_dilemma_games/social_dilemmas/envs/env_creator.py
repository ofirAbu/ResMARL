from social_dilemmas.envs.cleanup import CleanupEnv, CleanupEnvWithMessagesSelf, CleanupEnvWithMessagesGlobal
from social_dilemmas.envs.harvest import HarvestEnv, HarvestEnvWithMessagesSelf, HarvestEnvWithMessagesGlobal
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

    elif env == "harvest_msg_self":

        def env_creator(_):
            return HarvestEnvWithMessagesSelf(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
            )

    elif env == "harvest_msg_global":

        def env_creator(_):
            return HarvestEnvWithMessagesGlobal(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
            )

    elif env == "cleanup":

        def env_creator(_):
            return CleanupEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
            )

    elif env == "cleanup_msg_self":

        def env_creator(_):
            return CleanupEnvWithMessagesSelf(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
            )

    elif env == "cleanup_msg_global":

        def env_creator(_):
            return CleanupEnvWithMessagesGlobal(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
            )

    elif env == "switch":

        def env_creator(_):
            return SwitchEnv(num_agents=num_agents, args=args)

    return env_creator
