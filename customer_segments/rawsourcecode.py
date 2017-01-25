# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt

# Import supplementary visualizations code visuals.py
# import visuals as vs

# Pretty display for notebooks
# %matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

# Display a description of the dataset
display(data.describe())

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [16, 80, 59]

indices = [28, 3, 80]
import seaborn as sns
# Create a DataFrame of the chosen samples
sample1 = pd.DataFrame(data.loc[[28]], columns = data.keys()).reset_index(drop = True)
sample2 = pd.DataFrame(data.loc[[3]], columns = data.keys()).reset_index(drop = True)
sample3 = pd.DataFrame(data.loc[[80]], columns = data.keys()).reset_index(drop = True)

print "Chosen samples of wholesale customers dataset:"
display(samples)
# data.plot.box()

ax = sns.boxplot(data = data, palette="Set2", whis=np.inf)
sns.swarmplot(data = sample1, color="#2ecc71", size=26, marker='x')
sns.swarmplot(data = sample2, color=".25", size=6)
sns.swarmplot(data = sample3, color=".25", size=6)

colors = np.random.random((len(data),3))
markers = ['x','o','v','^','<']*100

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

from sklearn import cross_validation
from sklearn import tree
cols = list(data.columns)
for col in cols:
    droppedFeature = col
    feature_cols = data.columns.drop([droppedFeature])
    target_col = droppedFeature

    # Split the data into training and testing sets using the given feature as the target
    X_all = data[feature_cols]
    y_all = data[target_col]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.25, random_state=59)

    # Create a decision tree regressor and fit it to the training set
    regressor = tree.DecisionTreeRegressor(random_state=59)

    # Report the score of the prediction using the testing set
    regressor.fit(X_train, y_train)
    print col, ' can be predicted at ', regressor.score(X_test, y_test)


