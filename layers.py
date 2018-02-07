import tensorflow as tf
import numpy as np

#############################################################################################################
# Convolution Layer methods


def conv2d(name, inputs, kernel_size, padding, strides, out_channels, initializer):
    layer = tf.layers.Conv2D(
        out_channels,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=initializer,
        bias_initializer=tf.constant_initializer(0.0),
        padding=padding,
        name=name)
    preactivations = layer(inputs)
    activations = tf.nn.relu(preactivations)

    return preactivations, activations, (layer.kernel, layer.bias)


def dense(name, inputs, output_size, initializer):
    layer = tf.layers.Dense(
        output_size,
        kernel_initializer=initializer,
        bias_initializer=tf.constant_initializer(0.0),
        name=name)
    preactivations = layer(inputs)
    activations = tf.nn.tanh(preactivations)

    return preactivations, activations, (layer.kernel, layer.bias)


def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o


#############################################################################################################
# Pooling Layers methods

def max_pool_2d(x, size=(2, 2)):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, size_x, size_y, 1], padding='VALID',
                          name='pooling')


def upsample_2d(x, size=(2, 2)):
    """
    Bilinear Upsampling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :return: The output is the same input but doubled in both width and height (N,2H,2W,C).
    """
    h, w, _ = x.get_shape().as_list()[1:]
    size_x, size_y = size
    output_h = h * size_x
    output_w = w * size_y
    return tf.image.resize_bilinear(x, (output_h, output_w), align_corners=None, name='upsampling')


#############################################################################################################
# Utils for Layers methods

def variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    variable_summaries(w)
    return w


# Summaries for variables
def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def noise_and_argmax(logits):
    # Add noise then take the argmax
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


def openai_entropy(logits):
    # Entropy proposed by OpenAI in their A2C baseline
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def softmax_entropy(p0):
    # Normal information theory entropy by Shannon
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)


def mse(predicted, ground_truth):
    # Mean-squared error
    return tf.square(predicted - ground_truth) / 2.


def orthogonal_initializer(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # Orthogonal Initializer that uses SVD. The unused variables are just for passing in tensorflow
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init
