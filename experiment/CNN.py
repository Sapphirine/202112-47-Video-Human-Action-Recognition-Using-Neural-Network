import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import time


y_train = np.loadtxt("/Users/apple/Desktop/6893 final_project/database of experiment/UCI HAR Dataset/Train_data/Train_labels.txt")
x_train = np.loadtxt("/Users/apple/Desktop/6893 final_project/database of experiment/UCI HAR Dataset/Train_data/Train_data.txt")
x_train = np.reshape(x_train,(600,1,100,3))
y_train = np.reshape(y_train,(600,6))

y_test = np.loadtxt("/Users/apple/Desktop/6893 final_project/database of experiment/UCI HAR Dataset/Test_data/Test_labels.txt")
x_test = np.loadtxt("/Users/apple/Desktop/6893 final_project/database of experiment/UCI HAR Dataset/Test_data/Test_data.txt")
x_test = np.reshape(x_test,(60,1,100,3))
y_test = np.reshape(y_test,(60,6))

h_in = 1
w_in = 100
labels_out = 6
cha = 3

learning_rate = 0.0001
training_epochs = 500

def CNN(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        w1 = tf.get_variable("weight",[1,10,3,60],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable("bias",[60],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, w1, strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,1,20,1],strides=[1,1,2,1],padding="SAME")
    with tf.variable_scope('layer3-conv2'):
        w2 = tf.get_variable("weight",[1,6,60,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable("bias",[10],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, w2, strides=[1,1,1,1],padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
    with tf.variable_scope('layer4-fc1'):
        pool_shape = relu2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(relu2, [-1, nodes])

        w3 = tf.get_variable("weight",[nodes,1000],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(w3))
        b3 = tf.get_variable("bias", [1000], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.tanh(tf.matmul(reshaped, w3) + b3)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('layer5-fc2'):
        w4 = tf.get_variable("weight", [1000,6],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(w4))
        b4 = tf.get_variable("bias", [6], initializer=tf.constant_initializer(0.1))
        logits = tf.nn.softmax(tf.matmul(fc1, w4) + b4)
    return logits

X = tf.placeholder(tf.float32, shape=[None,h_in,w_in,cha])
Y = tf.placeholder(tf.float32, shape=[None,labels_out])
regularizer = tf.contrib.layers.l2_regularizer(0.001)
y_pre = CNN(X, False, regularizer)
loss = -tf.reduce_sum(Y*tf.log(y_pre))
loss2 =  tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_pre)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    tf.initialize_all_variables().run()
    for epoch in range(training_epochs):
        _, tra_, loss_output, pred_y = session.run([optimizer, accuracy, loss2, y_pre], feed_dict={X: x_train, Y: y_train})
        testa = session.run(accuracy, feed_dict={X: x_test, Y: y_test})
        print("Epoch: ", epoch, "Accuracy: ",session.run(accuracy, feed_dict={X: x_train, Y: y_train}))
        print(loss_output)
    confusion_matrix = tf.confusion_matrix(tf.argmax(y_train, 1), tf.argmax(pred_y, 1), num_classes=6)
    conf_numpy = confusion_matrix.eval()
    
import pandas as pd
conf_df = pd.DataFrame(conf_numpy)
import seaborn as sn
conf_fig = sn.heatmap(conf_df, annot=True, fmt="d", cmap="BuPu")
    
