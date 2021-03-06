import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import math
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def one_hot(integer_encoded):
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    return one_hot_encoded




####CNN####################################################
L1 = 16  # number of convolutions for first layer
L2 = 32  # number of convolutions for second layer
L3 = 128  # number of neurons for dense layer
learning_date = 0.0001  # learning rate
epochs = 20  # number of times we loop through training data
batch_size = 4  # number of data per batch
display_step = 1
loss_rec = np.zeros([epochs, 1])
training_eval = np.zeros([epochs, 2])

expression = np.loadtxt('output/ibd_otus.txt', dtype=float, delimiter=",")
label_vec = np.array(expression[:, -1], dtype=int)
expression = np.array(expression[:, :-1])
expression = normalize(expression, axis=1)
labels = []
for l in label_vec:
    if l == 1:
        labels.append([0, 1])
    else:
        labels.append([1, 0])
labels = np.array(labels, dtype=int)

# train/test data split
cut = int(0.8 * np.shape(expression)[0])
expression, labels = shuffle(expression, labels)
x_train = expression[:cut, :]
x_test = expression[cut:, :]
y_train = labels[:cut, :]
y_test = labels[cut:, :]

features = x_train.shape[1]
classes = y_train.shape[1]

xs = tf.placeholder(tf.float32, [None, features])
ys = tf.placeholder(tf.float32, [None, classes])
keep_prob = tf.placeholder(tf.float32)
x_shape = tf.reshape(xs, [-1, 1, features, 1])

# first conv
w_conv1 = weight_variable([3, 3, 1, L1])
b_conv1 = bias_variable([L1])
h_conv1 = tf.nn.relu(conv2d(x_shape, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second conv
w_conv2 = weight_variable([3, 3, L1, L2])
b_conv2 = bias_variable([L2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

tmp_shape = (int)(math.ceil(features / 4.0))
h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * tmp_shape * L2])

# third dense layer,full connected
w_fc1 = weight_variable([1 * tmp_shape * L2, L3])
b_fc1 = bias_variable([L3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fourth layer, output
w_fc2 = weight_variable([L3, classes])
b_fc2 = bias_variable([classes])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=ys))
optimizer = tf.train.AdamOptimizer(learning_date).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    total_batch = int(np.shape(x_train)[0] / batch_size)
    for epoch in range(epochs):
        avg_cost = 0.
        x_tmp, y_tmp = shuffle(x_train, y_train)
        for i in range(total_batch - 1):
            batch_x, batch_y = x_tmp[i * batch_size:i * batch_size + batch_size], \
                               y_tmp[i * batch_size:i * batch_size + batch_size]
            _, c, acc = sess.run([optimizer, cost, accuracy],
                                 feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.5})
            # Compute average loss
            avg_cost += c / total_batch

        del x_tmp
        del y_tmp

        ## Display logs per epoch step
        if epoch % display_step == 0:
            loss_rec[epoch] = avg_cost
            acc, y_s = sess.run([accuracy, y_conv], feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
            auc = roc_auc_score(y_train, y_s)
            training_eval[epoch] = [acc, auc]
            print("Epoch:", '%d' % (epoch + 1), "cost =", "{:.9f}".format(avg_cost),
                  "Training accuracy:", round(acc, 3), " Training auc:", round(auc, 3))

        if avg_cost <= 0.1:
            print("Early stopping.")
            break
    y_pred = y_conv.eval(feed_dict={xs: x_test, ys: y_test, keep_prob: 1.0})

    y_pred = np.argmax(np.where(y_pred > 0.5, 1, 0), axis=1)
    y_test = np.argmax(y_test, axis=1)  # one hot to int
    acc = round(accuracy_score(y_test, y_pred), 3)
    auc = round(roc_auc_score(y_test, y_pred), 3)

    f1 = round(f1_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)

    print("Testing accuracy: ", acc, " Testing auc: ", auc, " Testing f1: ",
          f1, " Testing precision: ", precision, " Testing recall: ", recall)
