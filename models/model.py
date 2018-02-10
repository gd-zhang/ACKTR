import tensorflow as tf
from layers import mse, openai_entropy
from utils.utils import find_trainable_variables
from ops import layer_collection as lc
from ops import optimizer as opt


class Model:
    def __init__(self, sess,
                 entropy_coef=0.01, value_function_coeff=0.5, max_gradient_norm=0.5,
                 optimizer_params=None, args=None):
        self.train_batch_size = args.num_envs * args.unroll_time_steps
        self.num_steps = args.unroll_time_steps
        self.num_stack = args.num_stack
        self.actions = None
        self.advantage = None
        self.reward = None
        self.keep_prob = None
        self.is_training = None
        self.step_policy = None
        self.train_policy = None
        self.learning_rate_decayed = None
        self.initial_state = None
        self.X_input_step_shape = None
        self.X_input_train_shape = None
        self.policy_gradient_loss = None
        self.value_function_loss = None
        self.optimize = None
        self.entropy = None
        self.loss = None
        self.learning_rate = None
        self.num_actions = None
        self.img_height, self.img_width, self.num_classes = None, None, None

        self.policy = Model.policy_name_parser(args.policy_class)
        self.sess = sess
        self.vf_coeff = value_function_coeff
        self.entropy_coeff = entropy_coef
        self.max_grad_norm = max_gradient_norm

        # optimization
        self.layer_collection = lc.LayerCollection()

        # TODO: cold start? check details in baselines/acktr
        # TODO: max_grad_norm or clip_kl
        self.initial_learning_rate = optimizer_params['learning_rate']
        self.moving_average = optimizer_params['moving_average']
        self.damping = optimizer_params['damping']
        self.kl_clip = optimizer_params['kl_clip']
        self.momentum = optimizer_params['momentum']

        self.cov_update_op = None
        self.inv_update_op = None
        self.factors = None

    def __set_action_space_params(self, num_actions):
        self.num_actions = num_actions

    def __set_observation_space_params(self, observation_space_params):
        self.img_height, self.img_width, self.num_classes = observation_space_params

    def init_input(self):
        with tf.name_scope('input'):
            self.X_input_train_shape = (
                None, self.img_height, self.img_width, self.num_classes * self.num_stack)
            self.X_input_step_shape = (
                None, self.img_height, self.img_width,
                self.num_classes * self.num_stack)

            self.actions = tf.placeholder(tf.int32, [None])  # actions
            self.advantage = tf.placeholder(tf.float32, [None])  # advantage function
            self.reward = tf.placeholder(tf.float32, [None])  # reward
            self.learning_rate = tf.placeholder(tf.float32, [])  # learning rate
            # is_training is for BN, we don't add BN by default
            self.is_training = tf.placeholder(tf.bool)  # is_training

    def init_network(self):
        # The model structure
        # TODO: how to handle reuse, it would register loss two times
        self.train_policy = self.policy(self.sess, self.X_input_train_shape, self.num_actions,
                                        self.layer_collection, reuse=False)

        self.step_policy = self.policy(self.sess, self.X_input_step_shape, self.num_actions,
                                       reuse=True)

        with tf.variable_scope('train_output'):
            negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.train_policy.policy_logits,
                labels=self.actions)
            self.policy_gradient_loss = tf.reduce_mean(self.advantage * negative_log_prob_action)
            self.value_function_loss = tf.reduce_mean(mse(tf.squeeze(self.train_policy.value_function), self.reward))
            self.entropy = tf.reduce_mean(openai_entropy(self.train_policy.policy_logits))
            self.loss = self.policy_gradient_loss - self.entropy * self.entropy_coeff + self.value_function_loss * self.vf_coeff

            # Gradient Clipping
            params = find_trainable_variables("policy")
            grads = tf.gradients(self.loss, params)

            # Apply Gradients
            grads = list(zip(grads, params))
            optimizer = opt.KfacOptimizer(learning_rate=self.learning_rate, cov_ema_decay=self.moving_average,
                                          damping=self.damping, layer_collection=self.layer_collection,
                                          norm_constraint=self.kl_clip, momentum=self.momentum)
            self.optimize = optimizer.apply_gradients(grads)
            self.cov_update_op = optimizer.cov_update_op
            self.inv_update_op = optimizer.inv_update_op
            self.inv_update_dict = optimizer.inv_updates_dict
            self.factors = self.layer_collection.get_factors()

    def build(self, observation_space_params, num_actions):
        self.__set_observation_space_params(observation_space_params)
        self.__set_action_space_params(num_actions)
        self.init_input()
        self.init_network()

    @staticmethod
    def policy_name_parser(policy_name):
        from models.cnn_policy import CNNPolicy
        policy_to_class = {'CNNPolicy': CNNPolicy}

        if policy_name in policy_to_class:
            return policy_to_class[policy_name]
        raise ValueError("There is no policy with this name. Make sure that the policy exists.")
