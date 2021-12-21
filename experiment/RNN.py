import tensorflow as tf
import numpy as np
X = []
X_path1 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt'
X_path2 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt'
X_path3 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt'
X_path4 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt'
X_path5 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt'
X_path6 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt'
X_path7 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt'
X_path8 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt'
X_path9 = '/Users/apple/Downloads/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt'
#for signal_type_path in X_path:
file1 = open(X_path1, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file1]])
file1.close()
file2 = open(X_path2, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file2]])
file2.close()
file3 = open(X_path3, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file3]])
file3.close()
file4 = open(X_path4, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file4]])
file4.close()
file5 = open(X_path5, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file5]])
file5.close()
file6 = open(X_path6, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file6]])
file6.close()
file7 = open(X_path7, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file7]])
file7.close()
file8 = open(X_path8, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file8]])
file8.close()
file9 = open(X_path9, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file9]])
file9.close()
X_train = np.transpose(np.array(X), (1, 2, 0))

X = []
X_path1 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt'
X_path2 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt'
X_path3 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt'
X_path4 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt'
X_path5 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt'
X_path6 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt'
X_path7 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt'
X_path8 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt'
X_path9 = '/Users/apple/Downloads/UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt'
#for signal_type_path in X_path:
file1 = open(X_path1, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file1]])
file1.close()
file2 = open(X_path2, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file2]])
file2.close()
file3 = open(X_path3, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file3]])
file3.close()
file4 = open(X_path4, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file4]])
file4.close()
file5 = open(X_path5, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file5]])
file5.close()
file6 = open(X_path6, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file6]])
file6.close()
file7 = open(X_path7, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file7]])
file7.close()
file8 = open(X_path8, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file8]])
file8.close()
file9 = open(X_path9, 'r')
X.append([np.array(data, dtype=np.float32) for data in[row.replace('  ', ' ').strip().split(' ') for row in file9]])
file9.close()
X_test = np.transpose(np.array(X), (1, 2, 0))

Y_train_path = '/Users/apple/Downloads/UCI HAR Dataset/train/y_train.txt'
file = open(Y_train_path, 'r')
y_train = np.array([data for data in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)-1
file.close()
n = int(np.max(y_train)) + 1
y_train = np.eye(n)[np.array(y_train.reshape(len(y_train)), dtype=np.int32)]

Y_test_path = '/Users/apple/Downloads/UCI HAR Dataset/test/y_test.txt'
file = open(Y_test_path, 'r')
y_test = np.array([data for data in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)-1
file.close()
n = int(np.max(y_test)) + 1
y_test = np.eye(n)[np.array(y_test.reshape(len(y_test)), dtype=np.int32)]

def RNN_Model(X):
    step_size = 128
    learning_rate = 0.005
    loss_amount = 0.005
    epoch = 200
    batch_size = 1500
    size_of_each_step = 9
    hidden_number = 32
    output_size = 6
    w_hidden = tf.Variable(tf.random_normal([size_of_each_step, hidden_number]))
    w_output = tf.Variable(tf.random_normal([hidden_number, output_size]))
    bias_hidden = tf.Variable(tf.random_normal([hidden_number], mean=3.0))
    bias_output = tf.Variable(tf.random_normal([output_size]))
    X = tf.transpose(X, [1, 0, 2])
    X = tf.reshape(X, [-1, size_of_each_step])
    X = tf.matmul(X, w_hidden) + bias_hidden
    X = tf.split(X, step_size, 0)
    #X = tf.reshape(X,[-1,128,hidden_number])
    
    layer1 = tf.contrib.rnn.BasicLSTMCell(hidden_number, forget_bias=0)
    layer2 = tf.contrib.rnn.BasicLSTMCell(hidden_number, forget_bias=0)
    layer3 = tf.contrib.rnn.BasicLSTMCell(hidden_number, forget_bias=0)
    layer4 = tf.contrib.rnn.MultiRNNCell([layer1, layer2, layer3])
    #layer4 = tf.contrib.rnn.MultiRNNCell([layer1, layer2])
    #init_state=layer4.zero_state(batch_size,dtype=tf.float32)
    
    #output,final_states = tf.nn.dynamic_rnn(layer4, X, initial_state=init_state, dtype=tf.float32)
    output,states = tf.contrib.rnn.static_rnn(layer4, X, dtype=tf.float32)
    print(np.array(output).shape)
    final_output = output[-1]
    pre = tf.matmul(final_output,w_output)+bias_output
    return pre

step_size = 128
size_of_each_step = 9
output_size = 6
X=tf.placeholder(tf.float32, shape=[None,step_size,size_of_each_step])
Y=tf.placeholder(tf.float32, shape=[None,output_size])
pred_Y = RNN_Model(X)
epoch = 10
count = len(X_train)
batch_size = 1500
sum_num = 0

loss_amount = 0.0025
learning_rate = 0.005
for i in tf.trainable_variables():
    sum_num = sum_num+tf.nn.l2_loss(i)
l2 = loss_amount * sum_num
#loss2 = loss_amount*sum_num
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
hings = tf.losses.hinge_loss(Y, pred_Y)
hings_loss = tf.reduce_mean(hings)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
#MSE_loss = tf.reduce_mean(tf.square(tf.cast(correct_pred, dtype=tf.float32)))
best_accuracy = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        for start, end in zip(range(0, count, batch_size), range(batch_size, count + 1, batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],Y: y_train[start:end]})
        pred_out, accuracy_out, loss_out, hingsloss = sess.run([pred_Y, accuracy, loss, hings_loss], feed_dict={X: X_test, Y: y_test})
        print('traing iter: {},'.format(i) + 'test accuracy: {},'.format(accuracy_out) + 'loss:{}'.format(loss_out)+'Hings_Loss:{}'.format(hingsloss))
        best_accuracy = max(best_accuracy, accuracy_out)
    confusion_matrix = tf.confusion_matrix(tf.argmax(y_test, 1), tf.argmax(pred_out, 1), num_classes=6)
    conf_numpy = confusion_matrix.eval()
#confusion matrix
import pandas as pd
conf_df = pd.DataFrame(conf_numpy)

import seaborn as sn
conf_fig = sn.heatmap(conf_df, annot=True, fmt="d", cmap="BuPu")
    
