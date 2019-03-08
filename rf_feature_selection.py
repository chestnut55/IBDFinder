import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import io_util


def rf_feature_selection(X, y):
    rf = RandomForestClassifier(random_state=0, n_estimators=200)

    cv = StratifiedKFold(n_splits=10, random_state=0)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []

    results = []

    for train_idx, test_idx in cv.split(X, y):
        x_train = X.ix[train_idx, :]
        x_test = X.ix[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        y_score = rf.predict_proba(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        y_score = np.argmax(y_score, axis=1)
        auc = roc_auc_score(y_test, y_score)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        results = pd.Series(importances, index=X.columns.values).sort_values(ascending=False).index.values

    print("accuracy: %0.3f (+/- %0.3f) "
          "& precision: %0.3f (+/- %0.3f) "
          "& recall: %0.3f (+/- %0.3f) "
          "& f1_macro: %0.3f (+/- %0.3f)" %
          (np.mean(accuracy_list), np.std(accuracy_list),
           np.mean(precision_list), np.std(precision_list),
           np.mean(recall_list), np.std(recall_list),
           np.mean(f1_list), np.std(f1_list)))

    df = pd.DataFrame(columns=['accuracy', 'auc', 'F1', 'precision', 'recall',
                               'accuracy_std', 'auc_std', 'f1_std', 'precision_std', 'recall_std'])

    for i in np.arange(10, 200, 5):
        sub_results = results[:i]
        accuracy = cross_val_score(rf, X.loc[:, sub_results], y, cv=cv, scoring='accuracy')
        auc = cross_val_score(rf, X.loc[:, sub_results], y, cv=cv, scoring='roc_auc')
        precision = cross_val_score(rf, X.loc[:, sub_results], y, cv=cv, scoring='precision')
        recall = cross_val_score(rf, X.loc[:, sub_results], y, cv=cv, scoring='recall')
        f1_macro = cross_val_score(rf, X.loc[:, sub_results], y, cv=cv, scoring='f1_macro')

        df.loc[len(df)] = accuracy.mean(), auc.mean(), f1_macro.mean(), precision.mean(), recall.mean(), \
                          accuracy.std(), auc.std(), f1_macro.std(), precision.std(), recall.std()

    df.to_csv('output/rf_feature_selection_result.csv')


if __name__ == "__main__":
    X, y, _left, _right = io_util.load()
    rf_feature_selection(X, y)
