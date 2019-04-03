import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.regularizers import l1, l2, l1_l2
from sklearn.metrics import roc_auc_score
import early_stop
import tensorflow as tf


import utils
from sparse_layer import Sparse


def gemlp(x_train, x_test, y_train, y_test, left, right):

    max_norm_constraint = keras.constraints.max_norm(3.)

    input = Input(shape=(np.shape(x_train)[1],), name='input')
    in_put = Dropout(0.1)(input)
    L1 = BatchNormalization()(in_put)
    L1 = Sparse(adjacency_mat=right,kernel_initializer='he_uniform',name='L1')(L1)
    # L1 = BatchNormalization()(L1) # don't use!
    L1 = Activation('relu')(L1)
    # L1 = LeakyReLU(alpha=0.3)(L1)
    L1 = Dropout(0.1)(L1)

    L2 = Dense(128,kernel_initializer='he_uniform',name='L2')(L1)
    L2 = BatchNormalization()(L2)
    L2 = Activation('relu')(L2)
    # L2 = LeakyReLU(alpha=0.3)(L2)
    L2 = Dropout(0.1)(L2)

    L3 = Dense(64,kernel_initializer='he_uniform',name='L3')(L2)
    L3 = BatchNormalization()(L3)
    L3 = Activation('relu')(L3)
    # L3 = LeakyReLU(alpha=0.3)(L3)
    L3 = Dropout(0.1)(L3)

    L4 = Dense(16,kernel_initializer='he_uniform',name='L4')(L3)
    L4 = BatchNormalization()(L4)
    L4 = Activation('relu')(L4)
    # L4 = LeakyReLU(alpha=0.3)(L4)
    L4 = Dropout(0.1)(L4)

    output = Dense(1,kernel_initializer='he_uniform', activation='sigmoid', name='output')(L4)

    model = Model(inputs=[input], outputs=[output])

    plot_model(model, to_file='gemlp_model.png', show_shapes=True)
    # ada = keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    adam = keras.optimizers.Adam(lr=0.001)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    earlyStopping = early_stop.LossCallBack(loss=0.1)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=200, verbose=0,callbacks=[earlyStopping],
                        batch_size=32)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    #
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

    y_predict = model.predict(x_test)

    left_embedding_layer_weights = model.layers[3].get_weights()[0]
    right_embedding_layer_weights = model.layers[6].get_weights()[0]

    gamma_c = 20
    gamma_numerator = np.sum(right, axis=0)
    gamma_denominator = np.sum(right, axis=0)
    gamma_numerator[np.where(gamma_numerator > gamma_c)] = gamma_c

    var_left = tf.reduce_sum(tf.abs(tf.multiply(left_embedding_layer_weights, right)), 0)
    var_right = tf.reduce_sum(tf.abs(right_embedding_layer_weights), 1)
    var_importance = tf.add(tf.multiply(tf.multiply(var_left, gamma_numerator), 1. / gamma_denominator),
                            tf.multiply(tf.multiply(var_right, gamma_numerator), 1. / gamma_denominator))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        var_imp = sess.run([var_importance])
        var_imp = np.reshape(var_imp, [np.shape(x_train)[1]])
        np.savetxt("output/l1_weights.txt", left_embedding_layer_weights, delimiter=",")
        np.savetxt("output/l2_weights.txt", right_embedding_layer_weights, delimiter=",")
        np.savetxt('output/var_ibd.csv', var_imp, delimiter=",")

    return y_predict, test_loss, test_acc


if __name__ == "__main__":
    X, y, left, right = utils.load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    y_predict, test_loss, test_acc = gemlp(X_train, X_test, y_train, y_test, right, right)

    roc_auc = roc_auc_score(y_test, y_predict)

    print('auc is ' + str(roc_auc), 'accuracy is ' + str(test_acc))

