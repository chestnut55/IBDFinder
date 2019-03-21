import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from nearest_neighbors import taxon_id_name
import utils
import matplotlib.pyplot as plt
import seaborn as sns


def analysis_metrics():
    var_ibd = np.loadtxt('output/var_ibd.csv', delimiter=",")

    df = pd.read_csv('output/ibd_otus.txt', sep='\t', index_col=0, header=0)
    nodes = df.columns.values.tolist()[:-1]

    results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values
    # feature selection
    y = df['label'].values
    X = df.drop(columns=['label'])

    rf = RandomForestClassifier(random_state=0, n_estimators=200)
    # svm = SVC(kernel='linear',probability=True)

    cv = StratifiedKFold(n_splits=5, random_state=0)

    df = pd.DataFrame(columns=['Accuracy', 'AUC', 'Precision', 'Recall',
                               'accuracy_std', 'auc_std', 'precision_std', 'recall_std'])

    for i in np.arange(10, 200, 5):
        results200 = results[:i]
        accuracy = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='accuracy')
        auc = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='roc_auc')
        precision = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='precision')
        recall = cross_val_score(rf, X.loc[:, results200], y, cv=cv, scoring='recall')
        # f1_macro = cross_val_score(svm, X.loc[:, results200], y, cv=cv, scoring='f1_macro')

        df.loc[len(df)] = accuracy.mean(), auc.mean(), precision.mean(), recall.mean(), \
                          accuracy.std(), auc.std(), precision.std(), recall.std()

        print("accuracy: %0.3f (+/- %0.3f) "
              "& precision: %0.3f (+/- %0.3f) "
              "& recall: %0.3f (+/- %0.3f) " %
              (accuracy.mean(), accuracy.std() * 2,
               precision.mean(), precision.std() * 2,
               recall.mean(), recall.std() * 2))

    df.to_csv('output/feature_selection_result.csv')


def calculate_cophenetic_distance(species_list):
    taxa_mapping_file = taxon_id_name()
    df = pd.read_csv('output/tips_distances.csv', sep='\t', index_col=0)
    taxa_names = [taxa_mapping_file[name.strip()] for name in df.columns.values]

    mx_cophenetic_distance = pd.DataFrame(df.values, columns=taxa_names, index=taxa_names)

    cols = mx_cophenetic_distance.columns.intersection(species_list)
    idx = mx_cophenetic_distance.index.intersection(species_list)
    result = mx_cophenetic_distance[cols].loc[idx, :]

    return np.sum(result.values)


def plot_cophenetic_distance():
    var_ibd = np.loadtxt('output/var_ibd.csv', delimiter=",")
    df = pd.read_csv('output/ibd_otus.txt', sep='\t', index_col=0, header=0)
    nodes = df.columns.values.tolist()[:-1]
    gedfn_results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values

    X, y, _left, _right = utils.load()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    rf_results = utils.rf_ranked_feature_selection(X_train, y_train)

    narr = []
    ged = []
    rf = []
    for i in np.arange(10, 110, 10):
        v1 = calculate_cophenetic_distance(gedfn_results[:i])
        v2 = calculate_cophenetic_distance(rf_results[:i])

        ged.append(v1)
        rf.append(v2)

    narr.append(ged)
    narr.append(rf)
    df = pd.DataFrame(np.asarray(narr).T, columns=['GEDFN', 'Random Forest'], index=np.arange(10, 110, 10))

    # sns.palplot(sns.color_palette("muted", 10))
    colors = sns.color_palette("hls", 8)
    ax = df[['GEDFN', 'Random Forest']].plot(title="Cophenetic distance", figsize=(9, 7), legend=True,
                                            fontsize=12, color=[colors[0],colors[2]])
    ax.set_xlabel("#Features", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    plt.show()


if __name__ == "__main__":
    # analysis_metrics()
    plot_cophenetic_distance()
