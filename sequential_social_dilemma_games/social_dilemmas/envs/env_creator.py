from social_dilemmas.envs.cleanup import CleanupEnv, CleanupEnvWithMessagesSelf, CleanupEnvWithMessagesGlobal
from social_dilemmas.envs.envs_with_perturbations.cleanup_with_perts import CleanupPerturbationsEnv, \
    CleanupPerturbationsEnvWithMessagesSelf, CleanupPerturbationsEnvWithMessagesGlobal
from social_dilemmas.envs.envs_with_perturbations.harvest_with_perts import HarvestPerturbationEnv, \
    HarvestPerturbationsEnvWithMessagesSelf, HarvestPerturbationsEnvWithMessagesGlobal
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

    # pert = 50
    # cleanup
    elif env == "cleanup_pert_50":

        def env_creator(_):
            return CleanupPerturbationsEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                perturbation_magnitude=50,
            )

    elif env == "cleanup_pert_50_msg_self":

        def env_creator(_):
            return CleanupPerturbationsEnvWithMessagesSelf(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=50,
            )

    elif env == "cleanup_pert_50_msg_global":

        def env_creator(_):
            return CleanupPerturbationsEnvWithMessagesGlobal(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=50,
            )

    # harvest
    elif env == "harvest_pert_50":

        def env_creator(_):
            return HarvestPerturbationEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                perturbation_magnitude=50,
            )

    elif env == "harvest_pert_50_msg_self":

        def env_creator(_):
            return HarvestPerturbationsEnvWithMessagesSelf(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=50,
            )

    elif env == "harvest_pert_50_msg_global":

        def env_creator(_):
            return HarvestPerturbationsEnvWithMessagesGlobal(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=50,
            )

    # pert = 150
    # cleanup
    elif env == "cleanup_pert_150":

        def env_creator(_):
            return CleanupPerturbationsEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                perturbation_magnitude=150,
            )

    elif env == "cleanup_pert_150_msg_self":

        def env_creator(_):
            return CleanupPerturbationsEnvWithMessagesSelf(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=150,
            )

    elif env == "cleanup_pert_150_msg_global":

        def env_creator(_):
            return CleanupPerturbationsEnvWithMessagesGlobal(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=150,
            )

    # harvest
    elif env == "harvest_pert_150":

        def env_creator(_):
            return HarvestPerturbationEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                perturbation_magnitude=150,
            )

    elif env == "harvest_pert_150_msg_self":

        def env_creator(_):
            return HarvestPerturbationsEnvWithMessagesSelf(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=150,
            )

    elif env == "harvest_pert_150_msg_global":

        def env_creator(_):
            return HarvestPerturbationsEnvWithMessagesGlobal(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=150,
            )

    # pert = 200
    # cleanup
    elif env == "cleanup_pert_200":

        def env_creator(_):
            return CleanupPerturbationsEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                perturbation_magnitude=200,
            )

    elif env == "cleanup_pert_200_msg_self":

        def env_creator(_):
            return CleanupPerturbationsEnvWithMessagesSelf(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=200,
            )

    elif env == "cleanup_pert_200_msg_global":

        def env_creator(_):
            return CleanupPerturbationsEnvWithMessagesGlobal(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=200,
            )

    # harvest
    elif env == "harvest_pert_200":

        def env_creator(_):
            return HarvestPerturbationEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                perturbation_magnitude=200,
            )

    elif env == "harvest_pert_200_msg_self":

        def env_creator(_):
            return HarvestPerturbationsEnvWithMessagesSelf(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=200,
            )

    elif env == "harvest_pert_200_msg_global":

        def env_creator(_):
            return HarvestPerturbationsEnvWithMessagesGlobal(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                use_messages_attribute=True,
                perturbation_magnitude=200,
            )


    elif env == "switch":

        def env_creator(_):
            return SwitchEnv(num_agents=num_agents, args=args)

    return env_creator
