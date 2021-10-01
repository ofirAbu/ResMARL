from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from config.constants import CONFUSION_UPPER_BOUND
from models.actor_critic_lstm import ActorCriticLSTM
from models.common_layers import build_conv_layers, build_fc_layers
from models.next_reward_lstm import NextRewardLSTM

tf = try_import_tf()


class MessagesWithSelfConfusionModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        A base model that uses messages to reduce its self confusion.

        :param obs_space: The observation space shape.
        :param action_space: The amount of available actions to this agent.
        :param num_outputs: The amount of available actions to this agent.
        :param model_config: The model config dict. Used to determine size of conv and fc layers.
        :param name: The model name.
        """
        super(MessagesWithSelfConfusionModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self._other_agent_actions = None
        self._visibility = None
        self._intrinsic_reward = None

        self.obs_space = obs_space
        self.num_outputs = num_outputs
        self.actions_num_outputs = int((num_outputs - CONFUSION_UPPER_BOUND) / 2)
        self.messages_num_outputs = int((num_outputs - CONFUSION_UPPER_BOUND) / 2)

        self.num_other_agents = model_config["custom_options"]["num_other_agents"]
        self.influence_divergence_measure = model_config["custom_options"][
            "influence_divergence_measure"
        ]
        self.influence_only_when_visible = model_config["custom_options"][
            "influence_only_when_visible"
        ]

        self.encoder_model = self.create_messages_model_encoder(obs_space, model_config)

        self.register_variables(self.encoder_model.variables)
        self.encoder_model.summary()

        inner_obs_space = self.encoder_model.output_shape[-1]
        # Action selection/value function
        cell_size = model_config["custom_options"].get("cell_size")

        inner_obs_space_with_messages = inner_obs_space + self.num_other_agents
        inner_obs_space_with_messages_and_confusion = inner_obs_space + self.num_other_agents * 2

        # note that action space is [action, message, conf_level]
        self.actions_policy_model = ActorCriticLSTM(
            inner_obs_space_with_messages,
            action_space[0],
            self.actions_num_outputs,
            model_config,
            "actions_policy",
            cell_size=cell_size,
        )
        self.messages_policy_model = ActorCriticLSTM(
            inner_obs_space_with_messages_and_confusion,
            action_space[1],
            self.messages_num_outputs,
            model_config,
            "messages_policy",
            cell_size=cell_size,
        )
        self.next_reward_prediction_model = NextRewardLSTM(
            inner_obs_space,
            action_space[1],
            1,
            model_config,
            "next_reward_model",
            cell_size=cell_size,
        )

        self.register_variables(self.actions_policy_model.rnn_model.variables)
        self.actions_policy_model.rnn_model.summary()

        self.register_variables(self.messages_policy_model.rnn_model.variables)
        self.messages_policy_model.rnn_model.summary()

        self.register_variables(self.next_reward_prediction_model.rnn_model.variables)
        self.next_reward_prediction_model.rnn_model.summary()

    @staticmethod
    def create_messages_model_encoder(obs_space, model_config):
        """
        Creates the convolutional part of the mesages mode, has two output heads, one for the messages and one for the
        actions.
        Also casts the input uint8 observations to float32 and normalizes them to the range [0,1].
        :param obs_space: The agent's observation space.
        :param model_config: The config dict containing parameters for the convolution type/shape.
        :return: A new Model object containing the convolution.
        """
        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        # Determine vision network input shape: add an extra none for the time dimension
        inputs = tf.keras.layers.Input(shape=original_obs_dims, name="observations", dtype=tf.uint8)

        # Divide by 255 to transform [0,255] uint8 rgb pixel values to [0,1] float32.
        last_layer = tf.keras.backend.cast(inputs, tf.float32)
        last_layer = tf.math.divide(last_layer, 255.0)

        # Build the CNN layers
        conv_out = build_conv_layers(model_config, last_layer)

        # Add the fully connected layers
        last_layer = build_fc_layers(model_config, conv_out, "actions_policy")

        return tf.keras.Model(inputs, [last_layer], name="Baseline_Encoder_Model")

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Evaluate the model.
        Adds time dimension to batch before sending inputs to forward_rnn()
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The policy logits and state.
        """
        self._other_agent_actions = input_dict["obs"]["other_agent_actions"]
        self._visibility = input_dict["obs"]["visible_agents"]

        ac_critic_encoded_obs = self.encoder_model(inputs=input_dict["obs"]["curr_obs"])
        rnn_input_dict = {
            "ac_trunk": ac_critic_encoded_obs,
            "other_agent_actions": input_dict["obs"]["other_agent_actions"],
            "visible_agents": input_dict["obs"]["visible_agents"],
            "prev_actions": input_dict["prev_actions"],
            "other_agent_messages": input_dict["obs"]["other_agent_messages"],
            "other_agent_predicted_rewards": input_dict["obs"]["other_agent_predicted_rewards"],
        }

        # Add time dimension to rnn inputs
        for k, v in rnn_input_dict.items():
            rnn_input_dict[k] = add_time_dimension(v, seq_lens)

        output, new_state = self.forward_rnn(input_dict=rnn_input_dict,
                                             state=state,
                                             seq_lens=seq_lens)
        # output here should've values for (actions, messages, self_confusion)
        # TODO: need to remember to replace by calculated self_confusion value
        # TODO: need to debug and understand state list
        self.compute_intrinsic_reward(input_dict)

        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        """
        Forward pass through the LSTM.
        Implicitly assigns the value function output to self_value_out, and does not return this.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The policy logits and new state.
        """
        # 1 - actions, 2 - messages, 3 - reward predictor
        h1, c1, h2, c2, h3, c3, *_ = state

        # Compute the next action
        ac_pass_dict = {"curr_obs": input_dict["ac_trunk"]}
        (self._actions_model_out, self._actions_value_out, output_h1,
         output_c1,) = self.actions_policy_model.forward_rnn(
            ac_pass_dict, [h1, c1], seq_lens
        )

        (self._messages_model_out, self._messages_value_out, output_h2,
         output_c2,) = self.actions_policy_model.forward_rnn(
            ac_pass_dict, [h2, c2], seq_lens
        )

        other_messages = input_dict["obs"]["other_agent_messages"]
        agent_messages = tf.expand_dims(input_dict["prev_actions"][:, :, 1], axis=-1)
        all_messages = tf.concat([tf.cast(agent_messages, tf.uint8), other_messages], axis=-1,
                                 name="concat_true_messages")
        self._true_one_hot_messages = self._reshaped_as_one_hot(all_messages, self.messages_num_outputs,
                                                                "forward_one_hot_messages")

        reward_predictor_pass_dict = {
            "curr_obs": input_dict["curr_obs"],
            "one_hot_total_messages": self._true_one_hot_messages,
            "values_predicted": self._actions_model_out
        }

        self._next_reward_pred, output_h3, output_c3 = self.next_reward_prediction_model.forward_rnn(
            reward_predictor_pass_dict, [h3, c3], seq_lens
        )

        # computing counterfactual immediate reward assuming different messages
        counterfactuals_reward_predictions = []

        for i in range(self.messages_num_outputs):
            counterfactual_messages = tf.pad(
                other_messages, paddings=[[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=i
            )

            one_hot_counterfactual_messages = self._reshaped_as_one_hot(
                counterfactual_messages, self.messages_num_outputs, "messages_with_counterfactual_one_hot"
            )

            reward_predictor_dict = {
                "curr_obs": input_dict["curr_obs"],
                "other_agent_messages": one_hot_counterfactual_messages,
                "values_predicted": self._actions_model_out
            }

            counterfactual_next_reward_pred, _, _ = self.next_reward_prediction_model.forward_rnn(reward_predictor_dict,
                                                                                                  [h3, c3],
                                                                                                  seq_lens)
            counterfactuals_reward_predictions.append(tf.expand_dims(counterfactual_next_reward_pred, axis=-2))

        self._counterfactuals_reward_predictions = tf.concat(
            counterfactuals_reward_predictions, axis=-2, name="concat_counterfactuals_reward_preds"
        )

        self._other_agent_actions = input_dict["other_agent_actions"]
        self._visibility = input_dict["visible_agents"]

        self._model_out = tf.reshape(tf.concat([self._actions_model_out, self._messages_model_out], axis=-1),
                                     [-1, self.num_outputs])
        self._value_out = tf.reshape(tf.concat([self._actions_value_out, self._messages_value_out], axis=-1),
                                     [-1, self.num_outputs])

        return self._model_out, [output_h1, output_c1]

    def compute_intrinsic_reward(self, input_dict):
        """
        We have the actual reward + we have the estimated reward given our vector of messages.
        So given some predicted R we can calculate the achieved confusion.
        We define the negative of the reward (of the messages) to be the inverse of:
        The dist(l1, l2...) between the minimal confusion and the actual confusion.
        """
        prev_actions = input_dict["prev_actions"]
        prev_rewards = input_dict["prev_rewards"]
        counterfactual_rewards = self._counterfactual_rewards
        predicted_rewards = prev_actions[:, -1]
        actual_rewards = prev_rewards
        confusion_levels = tf.math.divide(tf.math.abs(predicted_rewards - actual_rewards), actual_rewards,
                                          name='actual_confusion_levels')

        counterfactual_confusion_levels_preds = [
            tf.math.divide(tf.math.abs(counterfactual_reward - actual_rewards), actual_rewards,
                           name=f'hypothetical_confusion_levels_{i}') for i, counterfactual_reward in
            enumerate(counterfactual_rewards)]
        min_counterfactual_confusion_levels_preds = tf.reduce_min(tf.stack(counterfactual_confusion_levels_preds),
                                                                  axis=0)
        self._intrinsic_reward = -tf.norm(tf.abs(confusion_levels - min_counterfactual_confusion_levels_preds),
                                          ord="euclidean")

    @staticmethod
    def _reshaped_as_one_hot(raw_tensor, encoding_vector_length, name):
        """
        Converts the collection of vectors from a number encoding to a one-hot encoding.
        Then, flattens the one-hot encoding so that all concatenated one-hot vectors are the same
        dimension. E.g. with a num_outputs (action_space.n) of 3:
        _reshaped_one_hot_actions([0,1,2]) returns [1,0,0,0,1,0,0,0,1]
        :param raw_tensor: The tensor containing numeric encoding.
        :return: Tensor containing one-hot reshaped action values.
        """
        one_hot_encoding = tf.keras.backend.one_hot(indices=raw_tensor, num_classes=encoding_vector_length)

        # Extract partially known tensor shape and combine with "vector"_layer known shape
        # This combination is a bit contrived for a reason: the shape cannot be determined otherwise
        batch_time_dims = [
            tf.shape(one_hot_encoding)[k] for k in range(one_hot_encoding.shape.rank - 2)
        ]
        reshape_dims = batch_time_dims + [raw_tensor.shape[-1] * encoding_vector_length]
        reshaped = tf.reshape(one_hot_encoding, shape=reshape_dims, name=name)
        return reshaped

    def action_logits(self):
        """
        :return: The action logits from the latest forward pass.
        """
        return tf.reshape(self._actions_model_out, [-1])

    def value_function(self):
        """
        :return: The value function result from the latest forward pass.
        """
        return tf.reshape(self._actions_value_out, [-1])

    def counterfactual_predicted_rewards(self):
        return self._counterfactuals_reward_predictions

    def messages_intrinsic_reward(self):
        return self._intrinsic_reward

    def predicted_next_reward(self):
        return self._next_reward_pred

    def visibility(self):
        return tf.reshape(self._visibility, [-1, self.num_other_agents])

    def other_agent_actions(self):
        return tf.reshape(self._other_agent_actions, [-1, self.num_other_agents])

    @override(ModelV2)
    def get_initial_state(self):
        """
        :return: Initial state of this model. This model only has LSTM state from the policy_model.
        """
        return self.actions_policy_model.get_initial_state() + self.messages_policy_model.get_initial_state() + \
               self.next_reward_prediction_model.get_initial_state()
