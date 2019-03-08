import pandas as pd
import matplotlib.pyplot as plt


def plot():
    df = pd.read_csv('output/rf_feature_selection_result.csv', header=0, sep=',', index_col=0)

    _df = df[['accuracy', 'auc', 'F1', 'precision', 'recall']]

    _df.plot(linestyle='-')

    plt.show()


if __name__ == "__main__":
    plot()
