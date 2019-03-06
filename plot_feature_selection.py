import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/feature_selection_result.csv', header=0, sep=',',index_col=0)

_df = df[['accuracy', 'auc', 'F1', 'precision', 'recall']]
print(_df)

_df.plot(linestyle='-')

plt.show()