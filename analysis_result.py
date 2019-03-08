import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


def analysis():
    var_ibd = np.loadtxt('output/var_ibd.csv', delimiter=",")

    df = pd.read_csv('output/ibd_otus.txt', sep='\t', index_col=0, header=0)
    nodes = df.columns.values.tolist()[:-1]

    results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values
    # feature selection
    y = df['label'].values
    X = df.drop(columns=['label'])

    rf = RandomForestClassifier(random_state=0, n_estimators=200)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)
    # acc = round(accuracy_score(y_test, y_pred), 3)
    # auc = round(roc_auc_score(y_test, y_pred), 3)
    #
    # f1 = round(f1_score(y_test, y_pred), 3)
    # precision = round(precision_score(y_test, y_pred), 3)
    # recall = round(recall_score(y_test, y_pred), 3)
    #
    # print("Testing accuracy: ", acc, " Testing auc: ", auc, " Testing f1: ",
    #       f1, " Testing precision: ", precision, " Testing recall: ", recall)


    cv = StratifiedKFold(n_splits=10, random_state=0)

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

    df = pd.DataFrame(columns=['accuracy', 'auc', 'F1', 'precision', 'recall',
                               'accuracy_std','auc_std','f1_std','precision_std','recall_std'])

    for i in np.arange(10, 200, 5):
        results200 = results[:i]
        accuracy = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='accuracy')
        auc = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='roc_auc')
        precision = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='precision')
        recall = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='recall')
        f1_macro = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='f1_macro')

        df.loc[len(df)] = accuracy.mean(), auc.mean(), f1_macro.mean(), precision.mean(), recall.mean(),\
                          accuracy.std(),auc.std(),f1_macro.std(),precision.std(),recall.std()

        print("accuracy: %0.3f (+/- %0.3f) "
              "& precision: %0.3f (+/- %0.3f) "
              "& recall: %0.3f (+/- %0.3f) "
              "& f1_macro: %0.3f (+/- %0.3f)" %
              (accuracy.mean(), accuracy.std() * 2,
               precision.mean(), precision.std() * 2,
               recall.mean(), recall.std() * 2,
               f1_macro.mean(), f1_macro.std() * 2))

    df.to_csv('output/feature_selection_result.csv')

if __name__ == "__main__":
    analysis()