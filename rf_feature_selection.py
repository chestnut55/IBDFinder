import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

import utils


def rf_feature_selection(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, stratify=y, random_state=0)

    results = utils.rf_ranked_feature_selection(X_train, y_train)
    df = pd.DataFrame(columns=['Accuracy', 'AUC', 'F1', 'Precision', 'Recall',
                               'accuracy_std', 'auc_std', 'f1_std', 'precision_std', 'recall_std'])

    rf = RandomForestClassifier(random_state=0, n_estimators=200)
    cv = StratifiedKFold(n_splits=5, random_state=0)
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
    X, y, _left, _right = utils.load()
    rf_feature_selection(X, y)
