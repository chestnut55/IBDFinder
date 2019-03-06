'''
Deep Forward Network
'''
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn import preprocessing

## load in data
expression = np.loadtxt('output/otus.csv', dtype=float, delimiter=",")
labels = np.array(expression[:, -1], dtype=int)
expression = np.array(expression[:, :-1])
expression = preprocessing.normalize(expression, axis=1)
cut = int(0.8 * np.shape(expression)[0])
expression, labels = shuffle(expression, labels)
x_train = expression[:cut, :]
x_test = expression[cut:, :]
y_train = labels[:cut]
y_test = labels[cut:]


optimizers = keras.optimizers.Adam(lr=0.001)
model = Sequential()
model.add(Dense(1024, input_dim=np.shape(x_train)[1], activation='relu',
                kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(64, activation='relu', kernel_initializer='random_uniform'))
model.add(Dense(16, activation='relu', kernel_initializer='random_uniform'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))
model.compile(loss='binary_crossentropy', optimizer=optimizers, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=4, verbose=1)
scores = model.evaluate(x_train, y_train)
print("\n%s:%.2f%%" % (model.metrics_names[1], scores[1] * 100))

y_pred = model.predict(x_test)

y_pred = [round(x) for x in y_pred ]
acc = round(accuracy_score(y_test, y_pred), 3)
auc = round(roc_auc_score(y_test, y_pred), 3)
f1 = round(f1_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)

print("Testing accuracy: ", acc, " Testing auc: ", auc, " Testing f1: ",
      f1, " Testing precision: ", precision, " Testing recall: ", recall)
