import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import forward

TRAINING_STEP=30000
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZER_RATE=0.0001
batch_size=100
MOVING_AVERAGE_DECAY=0.99

MODEL_SAVE_PATH=("E:/model/")
MODEL_NAME = "model.ckpt"

def train(mnist):

    x=tf.placeholder(tf.float32,[None,forward.INPUT_NODE],name='x_input')
    y=tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE],name='y_input')
    prediction=forward.inference(x,REGULARIZER_RATE)
    global_step = tf.Variable(0, trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=tf.argmax(y,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    #loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/batch_size,
        LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEP):
            xs,ys=mnist.train.next_batch(batch_size)
            _,loss_value,step=sess.run([train_op,cross_entropy_mean,global_step],feed_dict={x:xs,y:ys})
            if i%1000==0:
                print("After %d training step(s),loss on train batch is %g."%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
def main(argv=None):
        mnist=input_data.read_data_sets("E:/MNIST_data/",one_hot=True)
        train(mnist)

if __name__ == '__main__':
        tf.app.run()
