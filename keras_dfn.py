import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import utils


class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.01, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)
            exit()

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def dfn(x_train, x_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(np.shape(x_train)[1], input_dim=np.shape(x_train)[1], activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # ada = keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    es = EarlyStoppingByLossVal(monitor='val_loss',value=0.01, verbose=1)

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=50, verbose=1, callbacks=[es],
                        batch_size=16)

    train_score, train_acc = model.evaluate(x_train, y_train)
    test_score, test_acc = model.evaluate(x_test, y_test)

    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, y, left, right = utils.load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    dfn(X_train, X_test, y_train, y_test)
    # skf = StratifiedKFold(n_splits=10, random_state=0)
    # for train_idx, test_idx in skf.split(X, y):
    #     x_train, x_test, y_train, y_test = X.ix[train_idx, :], X.ix[test_idx, :], y[train_idx], y[test_idx]
    #     dfn(x_train, x_test, y_train, y_test)
    #     break
