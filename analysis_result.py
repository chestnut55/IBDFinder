import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

var_ibd = np.loadtxt('var_ibd.csv', delimiter=",")

df = pd.read_csv('output/ibd_otus.csv', sep='\t', index_col=0, header=0)
nodes = df.columns.values.tolist()[:-1]

results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values
# feature selection
y = df['label'].values
X = df.drop(columns=['label'])

rf = RandomForestClassifier(random_state=0, n_estimators=200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = round(accuracy_score(y_test, y_pred), 3)
auc = round(roc_auc_score(y_test, y_pred), 3)

f1 = round(f1_score(y_test, y_pred, average='macro'), 3)
precision = round(precision_score(y_test, y_pred, average='macro'), 3)
recall = round(recall_score(y_test, y_pred, average='macro'), 3)

print("Testing accuracy: ", acc, " Testing auc: ", auc, " Testing f1: ",
      f1, " Testing precision: ", precision, " Testing recall: ", recall)


# cv = StratifiedKFold(n_splits=5, random_state=0)

# accuracy = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
# precision = cross_val_score(rf, X, y, cv=cv, scoring='precision')
# recall = cross_val_score(rf, X, y, cv=cv, scoring='recall')
# f1_macro = cross_val_score(rf, X, y, cv=cv, scoring='f1_macro')
# print("accuracy: %0.3f (+/- %0.3f) "
#       "& precision: %0.3f (+/- %0.3f) "
#       "& recall: %0.3f (+/- %0.3f) "
#       "& f1_macro: %0.3f (+/- %0.3f)" %
#       (accuracy.mean(), accuracy.std() * 2,
#        precision.mean(), precision.std() * 2,
#        recall.mean(), recall.std() * 2,
#        f1_macro.mean(), f1_macro.std() * 2))

# results200 = results[:200]
# accuracy = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='accuracy')
# precision = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='precision')
# recall = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='recall')
# f1_macro = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='f1_macro')
# print("accuracy: %0.3f (+/- %0.3f) "
#       "& precision: %0.3f (+/- %0.3f) "
#       "& recall: %0.3f (+/- %0.3f) "
#       "& f1_macro: %0.3f (+/- %0.3f)" %
#       (accuracy.mean(), accuracy.std() * 2,
#        precision.mean(), precision.std() * 2,
#        recall.mean(), recall.std() * 2,
#        f1_macro.mean(), f1_macro.std() * 2))



