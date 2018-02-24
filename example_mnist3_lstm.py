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


#--------------------------------------Define Graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    #------------------------------------construct LSTM------------------------------------------#
    #place hoder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None, 28, 28),name="input_placeholder")
    # X_p = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="input_placeholder")
    y_p=tf.placeholder(dtype=tf.float32,shape=(None,10),name="pred_placeholder")

    #lstm instance
    lstm_forward_1 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_forward_2 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS2)
    lstm_forward = rnn.MultiRNNCell(cells=[lstm_forward_1,lstm_forward_2])

    lstm_backward_1 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_backward_2 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS2)
    lstm_backward = rnn.MultiRNNCell(cells=[lstm_backward_1,lstm_backward_2])

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_forward,
        cell_bw=lstm_backward,
        inputs=X_p,
        dtype=tf.float32
    )
    print outputs

    outputs_fw = outputs[0]
    # print outputs_fw
    outputs_bw = outputs[1]
    h = outputs_fw[:,-1,:] + outputs_bw[:,-1,:]
    # print(h.shape)

    #---------------------------------define loss and optimizer----------------------------------#
    cross_loss=tf.losses.softmax_cross_entropy(onehot_labels=y_p,logits=h)
    #print(loss.shape)

    correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y_p, 1))
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
                    X_p: map(lambda x: x[0], training_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE]),
                    y_p: map(lambda x: x[1], training_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
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
                X_p: map(lambda x: x[0], test_data),
                y_p: map(lambda x: x[1], test_data)
            }
        )
        test_losses.append(test_loss)
        test_accus.append(test_accuracy)
        print("average test loss:", sum(test_losses) / len(test_losses))
        print("test accuracy:", sum(test_accus)/len(test_accus))