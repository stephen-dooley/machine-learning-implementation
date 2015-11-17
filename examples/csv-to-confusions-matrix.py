from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

# est = LinearRegression(fit_intercept=False)
# # random training data
# X = np.random.rand(10, 2)
# y = np.random.randint(2, size=10)
# est.fit(X, y)
# est.coef_   # access coefficients

##########################################################################################################################
df = pd.read_csv('/Users/stephen/Downloads/Default.csv', index_col=0)

# downsample negative cases -- there are many more negatives than positives
indices = np.where(df.default == 'No')[0]
# if the argument is an integer, then a 1-D array filled with generated values is returned
rng = np.random.RandomState(1)
rng.shuffle(indices)
# get the number of default = "yes" === 333
n_pos = (df.default == 'Yes').sum()

# drop the default = "no" so that are 333 of yes and 333 of no
df = df.drop(df.index[indices[n_pos:]])
df.to_csv('/Users/stephen/Documents/college/machine-learning/assignment-3/data/test.csv');
df.head()


# get feature/predictor matrix as numpy array
X = df[['balance', 'income']].values

# encode class labels
classes, y = np.unique(df.default.values, return_inverse=True)
y = (y * 2) - 1  # map {0, 1} to {-1, 1}

##########################################################################################################################

# fit OLS regression
est = LinearRegression(fit_intercept=True, normalize=True)
est.fit(X, y)

# the larger operator will return a boolean array which we will cast as integers for fancy indexing
y_pred = (2 * (est.predict(X) > 0.0)) - 1

def confusion_matrix(y_test, y_pred):
    cm = sk_confusion_matrix(y, y_pred)
    cm = pd.DataFrame(data=cm, columns=[-1, 1], index=[-1, 1])
    cm.columns.name = 'Predicted label'
    cm.index.name = 'True label'
    error_rate = (y_pred != y).mean()
    print('error rate: %.2f' % error_rate)
    return cm

print('_______________________')
print(confusion_matrix(y, y_pred))
print('_______________________\n\n')
##########################################################################################################################

# create 80%-20% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit on training data
est = LinearRegression().fit(X_train, y_train)

# test on data that was not used for fitting
y_pred = (2 * (est.predict(X) > 0.0)) - 1

print('_______________________')
print(confusion_matrix(y_test, y_pred))
print('_______________________\n\n')

##########################################################################################################################
