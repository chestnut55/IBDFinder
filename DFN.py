import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
import utils

# deep feedforward network
def dfn(x_train, x_test, y_train, y_test):
    def multilayer_perceptron(x, weights, biases, keep_prob):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob=keep_prob)

        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        ## Do not use batch-norm
        # layer_3 = tf.contrib.layers.batch_norm(layer_3, center=True, scale=True,
        #                                   is_training=is_training)
        layer_3 = tf.nn.relu(layer_3)
        layer_3 = tf.nn.dropout(layer_3, keep_prob=keep_prob)

        out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
        return out_layer

    # tf.reset_default_graph()

    ## hyper-parameters and settings
    L2 = False
    learning_rate = 0.001
    training_epochs = 50
    batch_size = 32
    display_step = 1

    n_features = np.shape(x_train)[1]
    n_hidden_1 = 128
    n_hidden_2 = 64
    n_hidden_3 = 16
    n_classes = 2

    ## initiate training logs
    loss_rec = np.zeros([training_epochs, 1])
    training_eval = np.zeros([training_epochs, 2])

    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.int32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    weights = {
        'h1': tf.Variable(tf.truncated_normal(shape=[n_features, n_hidden_1], stddev=0.1)),
        'h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),
        'h3': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_hidden_3], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal(shape=[n_hidden_3, n_classes], stddev=0.1))

    }

    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'b3': tf.Variable(tf.zeros([n_hidden_3])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    if L2:
        reg = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
              tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['out'])
        cost = tf.reduce_mean(cost + 0.001 * reg)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    ## Evaluation
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    y_score = tf.nn.softmax(logits=pred)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(np.shape(x_train)[0] / batch_size)

        ## Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            x_tmp, y_tmp = shuffle(x_train, y_train)
            # x_tmp, y_tmp = x_train, y_train
            # Loop over all batches
            for i in range(total_batch - 1):
                batch_x, batch_y = x_tmp[i * batch_size:i * batch_size + batch_size], \
                                   y_tmp[i * batch_size:i * batch_size + batch_size]

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,
                                                              keep_prob: 0.5,
                                                              lr: learning_rate
                                                              })
                # Compute average loss
                avg_cost += c / total_batch

            del x_tmp
            del y_tmp

            ## Display logs per epoch step
            if epoch % display_step == 0:
                loss_rec[epoch] = avg_cost
                acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_train, y: y_train, keep_prob: 1})
                auc = metrics.roc_auc_score(y_train, y_s)
                training_eval[epoch] = [acc, auc]
                # print("Epoch:", '%d' % (epoch + 1), "cost =", "{:.9f}".format(avg_cost),
                #       "Training accuracy:", round(acc, 3), " Training auc:", round(auc, 3))

            if avg_cost < 0.1:
                print("Early stopping.")
                break

        ## Testing cycle
        acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_test, y: y_test, keep_prob: 1})

        auc = round(roc_auc_score(y_test, y_s), 3)

        y_pred = np.argmax(y_s, axis=1)
        y_test = np.argmax(y_test, axis=1)  # one hot to int
        f1 = round(f1_score(y_test, y_pred), 3)
        precision = round(precision_score(y_test, y_pred), 3)
        recall = round(recall_score(y_test, y_pred), 3)

        print("Deep Feedforward Network Testing accuracy: ", acc, " Testing auc: ", auc, " Testing f1: ",
              f1, " Testing precision: ", precision, " Testing recall: ", recall)

        return acc, auc, f1, precision, recall, y_s[:, 1]

if __name__ == "__main__":

    X, y, left, right = utils.load()

    skf = StratifiedKFold(n_splits=10, random_state=0)
    for train_idx, test_idx in skf.split(X, y):
        x_train, x_test, y_train, y_test = X.ix[train_idx, :], X.ix[test_idx, :], y[train_idx], y[test_idx]
        dfn(x_train, x_test, to_categorical(y_train), to_categorical(y_test))
        break