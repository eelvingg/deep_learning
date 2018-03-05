import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import gzip
import pickle


BATCH_SIZE = 100
HIDDEN_UNITS1 = 30
HIDDEN_UNITS2 = 10
LEARNING_RATE = 0.001
EPOCH = 50

TRAIN_EXAMPLES = 50000
# TEST_EXAMPLES=28000

#------------------------------------Generate Data-----------------------------------------------#

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
    # inputs = [np.reshape(x,(784)) for x in raw_data[0]]
    inputs = [np.reshape(x, (28, 28)) for x in raw_data[0]]
    labels = [vectorized_label(x) for x in raw_data[1]]
    return zip(inputs, labels)

def vectorized_label(j):
    e=np.zeros((10))
    e[j]=1.0
    return e

training_data, validation_data, test_data = data_wrapper()

#-----------------------------------------------------------------------------------------------------#

def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    weight_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable('W_layer', weight_shape, initializer=weight_init)
    b = tf.get_variable('b_layer', bias_shape, initializer=bias_init)

    return tf.nn.relu(tf.matmul(input, W) + b)


def conv2d(input, weight_shape, bias_shape):
    dim = weight_shape[0] * weight_shape[1] * weight_shape[2]

    weight_init = tf.random_normal_initializer(stddev=(2.0/dim)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)

    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')

    return tf.nn.relu(tf.nn.bias_add(conv_out, b))


def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')



def inference(x, keep_prob):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    with tf.variable_scope('conv_1'):
        conv_1 = conv2d(x, [5, 5, 1, 32], [32])
        pool_1 = max_pool(conv_1)

    with tf.variable_scope('conv_2'):
        conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
        pool_2 = max_pool(conv_2)

    with tf.variable_scope('fc'):
        pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
        fc_1 = layer(pool_2_flat, [7*7*64, 1024], [1024])

        # dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope('output'):
        output = layer(fc_1_drop, [1024, 10], [10])

    # print(output)
    return output

# def loss(output, y):
#     xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
#     loss = tf.reduce_mean(xentropy)
#
#     return loss



#--------------------------------------Define Graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    x = tf.placeholder(dtype=tf.float32,shape=(None, 28, 28),name="input_placeholder")
    y = tf.placeholder(dtype=tf.float32,shape=(None,10),name="pred_placeholder")

    output = inference(x, 0.5)

    #---------------------------------define loss and optimizer----------------------------------#
    cross_loss=tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
    #print(loss.shape)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=cross_loss)

    init = tf.global_variables_initializer()


#-------------------------------------------Define Session---------------------------------------#
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(1, EPOCH+1):
        #results = np.zeros(shape=(TEST_EXAMPLES, 10))
        train_losses = []
        accus = []
        test_losses = []
        test_accus = []
        print('---------------')
        print("epoch:",epoch)
        for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
            _, train_loss, accu = sess.run(
                fetches=(optimizer, cross_loss, accuracy),
                feed_dict={
                    x: map(lambda x: x[0], training_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE]),
                    y: map(lambda x: x[1], training_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
                }
            )
            train_losses.append(train_loss)
            accus.append(accu)
        print("average training loss:", sum(train_losses) / len(train_losses))
        print("accuracy:", sum(accus)/len(accus))
        print('-----')



        test_output, test_loss, test_accuracy = sess.run(
            fetches=(optimizer, cross_loss, accuracy),
            feed_dict={
                x: map(lambda x: x[0], test_data),
                y: map(lambda x: x[1], test_data)
            }
        )
        test_losses.append(test_loss)
        test_accus.append(test_accuracy)
        print("average test loss:", sum(test_losses) / len(test_losses))
        print("test accuracy:", sum(test_accus)/len(test_accus))