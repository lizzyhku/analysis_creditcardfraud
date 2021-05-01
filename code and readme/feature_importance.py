# -*- utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


warnings.filterwarnings(action='once')
pd.set_option('mode.chained_assignment', None)

credit_data = pd.read_csv("./data/creditcard.csv", encoding = "utf8")

# print(credit_data)

X= credit_data.drop(labels='Class',axis = 1) #Features
y=credit_data.loc[:,'Class'] #Response

X = np.array(X)
y = np.array(y)
print(X.shape)
# print(y.shape)
# print(y)

# Build a forest and compute the impurity-based feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()