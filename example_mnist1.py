import tensorflow as tf
import gzip
import pickle
import numpy as np


# ------logsitic regression example------
session = tf.Session()
learning_rate = 0.01
training_epoch = 1000
display_step = 10
weight_init = tf.random_uniform_initializer(minval=-1, maxval=1)
bias_init = tf.constant_initializer(value=0)

def load_data():
    file=gzip.open('mnist.pkl.gz','rb')
    training_data,validation_data,test_data=pickle.load(file)
    file.close()
    return training_data,validation_data,test_data


def data_wrapper():
    tr_d,va_d,te_d=load_data()

    training_data = zip_data(tr_d)
    validation_data = zip_data(va_d)
    test_data = zip_data(te_d)

    return training_data, validation_data, test_data

def zip_data(raw_data):
    inputs=[np.reshape(x,(784)) for x in raw_data[0]]
    labels=[vectorized_label(x) for x in raw_data[1]]
    return zip(inputs, labels)

def vectorized_label(j):
    e=np.zeros((10))
    e[j]=1.0
    return e

def inference(x):
    init = tf.constant_initializer(value=0)
    W = tf.get_variable('W', [784, 10], initializer=init)
    b = tf.get_variable('b', [10], initializer=init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)

    return output

def loss(output, y):
    dot_product = y * tf.log(output)
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1) # indices = 1: collapse rows into values
    loss = tf.reduce_mean(xentropy)

    return loss

def training(cost, global_step):
    tf.summary.scalar('cost', cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


with tf.Graph().as_default():
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float', [None, 10])

    output = inference(x)
    cost = loss(output, y)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step)

    eval_op = evaluate(output, y)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess = tf.Session()

    summary_writer = tf.summary.FileWriter('logistic_logs/', graph_def=sess.graph_def)

    init_op = tf.global_variables_initializer()

    sess.run(init_op)

    training_data, validation_data, test_data = data_wrapper()

    # training cycle
    for epoch in range(training_epoch):

        feed_dict = {
            x: map(lambda x:x[0], training_data),
            y: map(lambda x:x[1], training_data)
        }
        sess.run(train_op, feed_dict=feed_dict)

        # display logs per epoch step
        if epoch % display_step == 0:
            val_feed_dict = {
                x: map(lambda x:x[0], validation_data),
                y: map(lambda x:x[1], validation_data)
            }
            accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
            print 'Validation Error at epoch %s:' % epoch, (1 - accuracy)

            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, sess.run(global_step))

            saver.save(sess, 'logistic_log/model-checkpoint', global_step=global_step)

    print 'Optimization Finished!'

    test_feed_dict = {
        x: map(lambda x:x[0], test_data),
        y: map(lambda x:x[1], test_data)
    }
    accuracy = sess.run(eval_op, feed_dict=test_feed_dict)

    print 'Test Accuracy: ', accuracy
