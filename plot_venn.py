import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pymrmr

df = pd.read_csv('output/ibd_otus.csv', sep='\t', index_col=0, header=0)

# feature selection
y = df['label'].values
X = df.drop(columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# RFE
model = LogisticRegression()
rfe = RFE(model, 100, step=5)
fit = rfe.fit(X_train, y_train)
rfe_results = X.columns.values[fit.support_]

# mRMR
mRMR_df = df.drop(columns=['label'])
mRMR_df.insert(loc=0, column='class', value=y)
mRMR_results = pymrmr.mRMR(mRMR_df, 'MIQ', 100)

# random forest
rf = RandomForestClassifier(random_state=0, n_estimators=200)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
rf_results = pd.Series(importances, index=X.columns.values).sort_values(ascending=False).index.values

# graph embedding feedforward neural network
var_ibd = np.loadtxt('var_ibd.csv', delimiter=",")
df = pd.read_csv('output/ibd_otus.csv', sep='\t', index_col=0, header=0)
nodes = df.columns.values.tolist()[:-1]
gedfn_results = pd.Series(var_ibd, index=nodes).sort_values(ascending=False).index.values

fig, axes = plt.subplots(1, 3)

venn3([set(gedfn_results[:100]), set(rf_results[:100]), set(rfe_results)],
      set_labels=('GEDFN', 'Random Forest', 'RFE'), ax=axes[0])

venn3([set(gedfn_results[:100]), set(rf_results[:100]), set(mRMR_results)],
      set_labels=('GEDFN', 'Random Forest', 'mRMR'), ax=axes[1])

venn3([set(gedfn_results[:100]), set(rfe_results), set(mRMR_results)],
      set_labels=('GEDFN', 'RFE', 'mRMR'), ax=axes[2])

plt.savefig('venn.png', bbox_inches='tight')
plt.show()
