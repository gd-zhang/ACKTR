import numpy as np
import tensorflow as tf
from models.base_policy import BasePolicy
from layers import conv2d, flatten, dense
from layers import orthogonal_initializer, noise_and_argmax


class CNNPolicy(BasePolicy):
    def __init__(self, sess, input_shape, num_actions, layer_collection=None, reuse=False, name='train'):
        super().__init__(sess, reuse)
        self.initial_state = []
        with tf.name_scope(name + "policy_input"):
            self.X_input = tf.placeholder(tf.uint8, input_shape)
        with tf.variable_scope("policy", reuse=reuse):
            inputs = tf.cast(self.X_input, tf.float32) / 255.
            pre1, act1, param1 = conv2d('conv1', inputs, kernel_size=(8, 8),
                                        padding='VALID', strides=(4, 4), out_channels=32,
                                        initializer=orthogonal_initializer(np.sqrt(2)))

            pre2, act2, param2 = conv2d('conv2', act1, kernel_size=(4, 4), padding='VALID',
                                        strides=(2, 2), out_channels=64,
                                        initializer=orthogonal_initializer(np.sqrt(2)))

            # TODO: in original ACKTR paper, the third conv has 32 filters to save computation
            pre3, act3, param3 = conv2d('conv3', act2, kernel_size=(3, 3), padding='VALID',
                                        strides=(1, 1), out_channels=32,
                                        initializer=orthogonal_initializer(np.sqrt(2)))

            conv3_flattened = flatten(act3)

            pre4, act4, param4 = dense('fc4', conv3_flattened, output_size=512,
                                       initializer=orthogonal_initializer(np.sqrt(2)))

            self.policy_logits, _, paramp = dense('policy_logits', act4, output_size=num_actions,
                                                  initializer=orthogonal_initializer(np.sqrt(1.0)))

            self.value_function, _, paramv = dense('value_function', act4, output_size=1,
                                                   initializer=orthogonal_initializer(np.sqrt(1.0)))

            with tf.name_scope('value'):
                self.value_s = self.value_function[:, 0]

            with tf.name_scope('action'):
                self.action_s = noise_and_argmax(self.policy_logits)

            # register parameters. K-FAC need to know about the inputs,
            # outputs and parameters for each layer
            if layer_collection is not None:
                layer_collection.register_conv2d(param1, (1, 4, 4, 1), 'VALID', inputs, pre1)
                layer_collection.register_conv2d(param2, (1, 2, 2, 1), 'VALID', act1, pre2)
                layer_collection.register_conv2d(param3, (1, 1, 1, 1), 'VALID', act2, pre3)
                layer_collection.register_fully_connected(param4, conv3_flattened, pre4)
                layer_collection.register_fully_connected(paramp, act4, self.policy_logits)
                layer_collection.register_fully_connected(paramv, act4, self.value_function)

                # mse ==> var=1.0 (Gauss-Netwon)
                layer_collection.register_categorical_predictive_distribution(self.policy_logits, name="logits")
                layer_collection.register_normal_predictive_distribution(self.value_function, var=1.0, name="mean")



    def step(self, observation, *_args, **_kwargs):
        # Take a step using the model and return the predicted policy and value function
        action, value = self.sess.run([self.action_s, self.value_s], {self.X_input: observation})
        return action, value, []  # dummy state

    def value(self, observation, *_args, **_kwargs):
        # Return the predicted value function for a given observation
        return self.sess.run(self.value_s, {self.X_input: observation})
