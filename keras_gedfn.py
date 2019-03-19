import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Concatenate
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.utils import to_categorical
import utils
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import Model
from keras.utils import plot_model
from sparse_layer import Sparse
import keras


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
    sparse_layer = Sparse(adjacency_mat=left)
    A1 = Input(shape=(np.shape(x_train)[1],), name='input')

    A2 = Dense(128, activation='relu', name='fully-layer')(A1)
    A2 = Dropout(0.5)(A2)
    A2_hat = sparse_layer(A1)
    A2_hat = Dropout(0.5)(A2_hat)

    concat_layer = Concatenate()([A2, A2_hat])

    A3 = Dense(64, activation='relu', name='A3')(concat_layer)
    A3 = Dropout(0.5)(A3)
    A4 = Dense(16, activation='relu', name='A4')(A3)
    A4 = Dropout(0.5)(A4)
    A5 = Dense(1, activation='sigmoid', name='A5')(A4)

    model = Model(inputs=[A1], outputs=[A5])
    plot_model(model, to_file='model.png', show_shapes=True)

    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10)

    history = model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=16)
    print(history.history['loss'])

    return model.predict(x_test)


if __name__ == "__main__":
    X, y, left, right = utils.load()
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
        gedfn(X_train, X_test, y_train, y_test, left, right)
