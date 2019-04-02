import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Concatenate, LeakyReLU, BatchNormalization, Activation
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.utils import to_categorical
from keras.layers import BatchNormalization
import utils
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import Model
from keras.utils import plot_model
from sparse_layer import Sparse
import keras
from sklearn.metrics import roc_auc_score
import early_stop



def gedfn(x_train, x_test, y_train, y_test, left, right):
    '''
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param left: Specifies which inputs (rows) are connected to which outputs (columns)
    :param right:
    :return:
    '''
    left = left[:, ~np.all(left == 0, axis=1)]
    sparse_layer = Sparse(adjacency_mat=left,kernel_initializer='he_uniform')

    input = Input(shape=(np.shape(x_train)[1],))
    input_hat = Dropout(0.1)(input)
    input_batch_norm = BatchNormalization()(input_hat)

    h1_layer = Dense(128,kernel_initializer='he_uniform')(input_batch_norm)
    h1_layer = BatchNormalization()(h1_layer)
    h1_layer = Activation('relu')(h1_layer)
    # h1_layer = LeakyReLU(alpha=0.3)(h1_layer)
    h1_layer = Dropout(0.1)(h1_layer)

    h1_layer_hat = sparse_layer(input_batch_norm)
    h1_layer_hat = BatchNormalization()(h1_layer_hat)
    h1_layer_hat = Activation('relu')(h1_layer_hat)
    # h1_layer_hat = LeakyReLU(alpha=0.3)(h1_layer_hat)
    h1_layer_hat = Dropout(0.1)(h1_layer_hat)

    concat_layer = Concatenate()([h1_layer, h1_layer_hat])

    h2_layer = Dense(64,kernel_initializer='he_uniform')(concat_layer)
    h2_layer = BatchNormalization()(h2_layer)
    h2_layer = Activation('relu')(h2_layer)
    # h2_layer = LeakyReLU(alpha=0.3)(h2_layer)
    h2_layer = Dropout(0.1)(h2_layer)

    h3_layer = Dense(16,kernel_initializer='he_uniform')(h2_layer)
    h3_layer = BatchNormalization()(h3_layer)
    h3_layer = Activation('relu')(h3_layer)
    # h3_layer = LeakyReLU(alpha=0.3)(h3_layer)
    h3_layer = Dropout(0.1)(h3_layer)

    output = Dense(1,kernel_initializer='he_uniform',activation='sigmoid')(h3_layer)

    model = Model(inputs=[input], outputs=[output])
    plot_model(model, to_file='gedfn_model.png', show_shapes=True)

    optimizer = keras.optimizers.Adam(lr=0.001)
    # optimizer = keras.optimizers.SGD(lr=0.001, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # earlyStopping = early_stop.LossCallBack(loss=0.1)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), verbose=1,
                        batch_size=32, callbacks=[es])
    test_loss, test_acc = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)

    # plt.plot(history.history['acc'], label='train')
    # plt.plot(history.history['val_acc'], label='test')
    # plt.legend()
    # plt.show()

    auc = roc_auc_score(y_test, y_predict)
    print('auc is ' + str(auc))

    return y_predict, test_loss, test_acc


if __name__ == "__main__":
    X, y, left, right = utils.load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    gedfn(X_train, X_test, y_train, y_test, left, right)
