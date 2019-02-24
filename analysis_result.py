import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

var_ibd = np.loadtxt('var_ibd.csv', delimiter=",")

df = pd.read_csv('output/ibd_otus.csv', sep='\t', index_col=0, header=0)
nodes = df.columns.values.tolist()[:-1]

results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values
# feature selection
y = df['label'].values
X = df.drop(columns=['label'])

cv = StratifiedKFold(n_splits=5, random_state=0)
rf = RandomForestClassifier(random_state=0, n_estimators=200)

accuracy = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
precision = cross_val_score(rf, X, y, cv=cv, scoring='precision')
recall = cross_val_score(rf, X, y, cv=cv, scoring='recall')
f1_macro = cross_val_score(rf, X, y, cv=cv, scoring='f1_macro')
print("accuracy: %0.3f (+/- %0.3f) "
      "& precision: %0.3f (+/- %0.3f) "
      "& recall: %0.3f (+/- %0.3f) "
      "& f1_macro: %0.3f (+/- %0.3f)" %
      (accuracy.mean(), accuracy.std() * 2,
       precision.mean(), precision.std() * 2,
       recall.mean(), recall.std() * 2,
       f1_macro.mean(), f1_macro.std() * 2))

results200 = results[:200]
accuracy = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='accuracy')
precision = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='precision')
recall = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='recall')
f1_macro = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='f1_macro')
print("accuracy: %0.3f (+/- %0.3f) "
      "& precision: %0.3f (+/- %0.3f) "
      "& recall: %0.3f (+/- %0.3f) "
      "& f1_macro: %0.3f (+/- %0.3f)" %
      (accuracy.mean(), accuracy.std() * 2,
       precision.mean(), precision.std() * 2,
       recall.mean(), recall.std() * 2,
       f1_macro.mean(), f1_macro.std() * 2))



