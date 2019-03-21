import pandas as pd
import matplotlib.pyplot as plt


def plot():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    df_gedfn = pd.read_csv('output/feature_selection_result.csv', header=0, sep=',', index_col=0)
    _df_gedfn = df_gedfn[['Accuracy', 'AUC', 'Precision', 'Recall']]
    _df_gedfn.plot(linestyle='-', ax=axes[0])
    axes[0].set_ylim([0.7, 1])
    axes[0].set_xlabel('#feature')
    axes[0].set_title('GEDFN')

    df_rf = pd.read_csv('output/rf_feature_selection_result.csv', header=0, sep=',', index_col=0)
    _df_rf = df_rf[['Accuracy', 'AUC', 'Precision', 'Recall']]
    _df_rf.plot(linestyle='-', ax=axes[1])
    axes[1].set_ylim([0.7, 1])
    axes[1].set_xlabel('#feature')
    axes[1].set_title('Random Forest')

    plt.show()


if __name__ == "__main__":
    plot()
