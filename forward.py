import tensorflow as tf

INPUT_NODE=784
OUTPUT_NODE=10
LAYEL1_NODE=500


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(regularizer)(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights1 = get_weight_variable([INPUT_NODE, LAYEL1_NODE], regularizer)
        biases1 = tf.get_variable('biases1', [LAYEL1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

    with tf.variable_scope('layer2'):
        weights2 = get_weight_variable([LAYEL1_NODE, OUTPUT_NODE], regularizer)
        biases2 = tf.get_variable('biases2', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights2) + biases2

    return layer2