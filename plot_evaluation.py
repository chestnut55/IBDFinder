import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_eval():
    df = pd.read_csv('output/evaluation.csv', header=0, sep=',')

    evaluation = df.columns.values
    methods = ['dfn786', 'gedfn', 'Random Forest']

    # dfn_diag = df.iloc[0::4, ].reset_index(drop=True)
    dfn786 = df.iloc[0::3, ].reset_index(drop=True)
    gedfn = df.iloc[1::3, ].reset_index(drop=True)
    RF = df.iloc[2::3, ].reset_index(drop=True)

    fig, axes = plt.subplots(5, 1)
    for i, name in enumerate(evaluation):
        dd = pd.concat([dfn786[name], gedfn[name], RF[name]], axis=1)
        means = np.mean(dfn786[name].values), np.mean(gedfn[name].values), np.mean(RF[name].values)

        dd.columns = [method + " " + str(round(mean, 3)) for method, mean in zip(methods, means)]
        dd.plot(ax=axes[i], linestyle='--')
        axes[i].set_ylabel(name)
    plt.show()


if __name__ == "__main__":
    plot_eval()
