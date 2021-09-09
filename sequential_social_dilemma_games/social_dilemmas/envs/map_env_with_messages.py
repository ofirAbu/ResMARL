import numpy as np
from gym.spaces import Box, Dict
from ray.rllib import MultiAgentEnv

from social_dilemmas.envs.map_env import MapEnv


class MapEnvWithMessages(MapEnv):
    def __init__(
            self,
            ascii_map,
            extra_actions,
            view_len,
            num_agents=1,
            color_map=None,
            return_agent_actions=False,
            use_collective_reward=False,
    ):
        super().__init__(ascii_map=ascii_map,
                         extra_actions=extra_actions,
                         view_len=view_len,
                         num_agents=num_agents,
                         color_map=color_map,
                         return_agent_actions=return_agent_actions,
                         use_collective_reward=use_collective_reward)

    @property
    def observation_space(self):
        obs_space = super().observation_space.spaces
        obs_space = {
            **obs_space,
            "other_agent_messages": Box(
                low=0, high=len(self.all_actions), shape=(self.num_agents - 1,),
                dtype=np.uint8,
            )
        }
        obs_space = Dict(obs_space)
        # Change dtype so that ray can put all observations into one flat batch
        # with the correct dtype.
        # See DictFlatteningPreprocessor in ray/rllib/models/preprocessors.py.
        obs_space.dtype = np.uint8
        return obs_space

    def step(self, actions):
        """
        overwriting original function to take action and write down messages, different
        is that in current case actions are list of action and message.
        """
        self.beam_pos = []
        agent_actions = {}
        agent_messages = {agent_id: action_message_tuple[-1] for agent_id, action_message_tuple in actions.items()}

        actions = {agent_id: action_message_tuple[0] for agent_id, action_message_tuple in actions.items()}
        for agent_id, action in actions.items():
            agent_action = self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        # Remove agents from color map
        for agent in self.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            self.single_update_world_color_map(row, col, self.world_map[row, col])

        self.update_moves(agent_actions)

        for agent in self.agents.values():
            pos = agent.pos
            new_char = agent.consume(self.world_map[pos[0], pos[1]])
            self.single_update_map(pos[0], pos[1], new_char)

        # execute custom moves like firing
        self.update_custom_moves(agent_actions)

        # execute spawning events
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()
        # Add agents to color map
        for agent in self.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            # Firing beams have priority over agents and should cover them
            if self.world_map[row, col] not in [b"F", b"C"]:
                self.single_update_world_color_map(row, col, agent.get_char_id())

        observations = {}
        rewards = {}
        dones = {}
        info = {}
        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = self.color_view(agent)
            # concatenate on the prev_actions to the observations
            if self.return_agent_actions:
                prev_actions = np.array(
                    [actions[key] for key in sorted(actions.keys()) if key != agent.agent_id]
                ).astype(np.uint8)
                prev_messages = np.array(
                    [agent_messages[key] for key in sorted(agent_messages.keys()) if key != agent.agent_id]
                ).astype(np.uint8)
                visible_agents = self.find_visible_agents(agent.agent_id)
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "other_agent_actions": prev_actions,
                    "other_agent_messages": prev_messages,
                    "visible_agents": visible_agents,
                    "prev_visible_agents": agent.prev_visible_agents,
                }
                agent.prev_visible_agents = visible_agents
            else:
                observations[agent.agent_id] = {"curr_obs": rgb_arr}
            rewards[agent.agent_id] = agent.compute_reward()
            dones[agent.agent_id] = agent.get_done()

        if self.use_collective_reward:
            collective_reward = sum(rewards.values())
            for agent in rewards.keys():
                rewards[agent] = collective_reward

        dones["__all__"] = np.any(list(dones.values()))
        return observations, rewards, dones, info

    def reset(self):
        """
        Also need overwrite since it returns obseration and messages
        """
        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        observations = {}
        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = self.color_view(agent)
            # concatenate on the prev_actions to the observations
            if self.return_agent_actions:
                # No previous actions so just pass in "wait" action
                prev_actions = np.array([4 for _ in range(self.num_agents - 1)]).astype(np.uint8)
                prev_messages = np.array([0 for _ in range(self.num_agents - 1)]).astype(np.uint8)
                visible_agents = self.find_visible_agents(agent.agent_id)
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "other_agent_actions": prev_actions,
                    "other_agent_messages": prev_messages,
                    "visible_agents": visible_agents,
                    "prev_visible_agents": visible_agents,
                }
                agent.prev_visible_agents = visible_agents
            else:
                observations[agent.agent_id] = {"curr_obs": rgb_arr}
        return observations
