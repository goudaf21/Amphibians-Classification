# Fady Gouda, Abhi Jha, Griffin Noe, Utkrist P Thapa
# CSCI 297-a
# 10/20/20
# Final Project
# eda.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

data=pd.read_csv('amphibeans.csv')

# Rename the columns from their longform into shortform
data.rename(columns = {'Motorway': 'MV', 'Green frogs': 'L1', 'Brown frogs': 'L2', 'Common toad': 'L3',
                       'Fire-bellied toad': 'L4', 'Tree frog': 'L5', 'Common newt': 'L6', 'Great crested newt': 'L7'},
                       inplace = True)

# Specify the columns which have the classification labels
label_cols = ['L1','L2','L3','L4','L5','L6','L7']

# One-hot encode the categorical features
data = pd.get_dummies(data = data, columns = ['MV','VR','SUR1','SUR2','SUR3','UR','FR','MR','CR','TR'])

# Get the names of the data's features
cols = data.columns
features = ['SR', 'NR', 'OR', 'RR', 'BR', 'VR_0', 'VR_1', 'VR_2', 'VR_3', 'VR_4',
       'SUR1_1', 'SUR1_2', 'SUR1_4', 'SUR1_6', 'SUR1_7', 'SUR1_9', 'SUR1_10',
       'SUR1_14', 'SUR2_1', 'SUR2_2', 'SUR2_6', 'SUR2_7', 'SUR2_9', 'SUR2_10',
       'SUR2_11', 'SUR3_1', 'SUR3_2', 'SUR3_5', 'SUR3_6', 'SUR3_7', 'SUR3_9',
       'SUR3_10', 'SUR3_11', 'UR_0', 'UR_1', 'UR_3', 'FR_0', 'FR_1', 'FR_2',
       'FR_3', 'FR_4', 'MR_0', 'MR_1', 'MR_2', 'CR_1', 'CR_2', 'TR_1', 'TR_2',
       'TR_5', 'TR_7', 'TR_11', 'TR_12', 'TR_14', 'TR_15']

# Run a correlation heatmap on the result labels to see if they are highly correlated
cm = np.corrcoef(data[features].values.T)
hm = heatmap(cm, row_names=features, column_names=features)
plt.show()

cm = np.corrcoef(data[label_cols].values.T)
hm = heatmap(cm, row_names=label_cols, column_names=label_cols)
plt.show()

# Create a blank list for the classification feature
row_list = []

# Iterate through each row of the data
for index, rows in data.iterrows():

    # Create a 7-digit binary number that represents what animals are present in the area
    label_list = rows.L1 * (10**0) + rows.L2 * (10**1) + rows.L3 * (10**2) + rows.L4 * (10**3) + rows.L5 * (10**4) + rows.L6 * (10**5) + rows.L7 * (10**6)

    # Append the 7-digit number to the list
    row_list.append(label_list)

# Remove the redundant features and set it equal to a new dataframe
data2 = data.drop(label_cols, axis=1)
cols2 = data2.columns

# create mapping for pd.DataFrame.replace() from 7 digit label to single digit label
mapping = dict()
for i, key in enumerate(np.unique(row_list)):
    mapping[key] = i

# Create a new feature that has the label lists
data2['label_list'] = row_list

# Replace the 7 digit numbers with single digit labels
data2['label_list'].replace(mapping, inplace=True)

# Set the x and y for the second data frame
source_x = data2[cols2]
y = data2['label_list']

# standardize the dataset
sc = StandardScaler()
sc.fit(source_x)
x = pd.DataFrame(sc.transform(source_x))

# Split the data at 80/20 train/test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("Original shape of x-train:", x_train.shape)

### PCA

#run PCA for a list of components, in order to decide the highest variance added by each eigen value
for i in range (20):
    pcatest= PCA(n_components=i)
    pcatest.fit(x_train,y_train)
    # print(pcatest.explained_variance_)

# After running the for loop, it seems that n_components=3 would provide the highest variance,
# and any component more than that does not add as much variance
pca=PCA(n_components=3)
pca.fit(x_train,y_train)
x_train_pca= pca.transform (x_train)
print("Post-PCA shape of x-train:", x_train_pca.shape)

#Here I try to show the graph when using n_components =2
# pcagraph=PCA(n_components=2)
# pcagraph.fit(x_train,y_train)
# x_train_graph=pca.transform (x_train)
# principleDF= pd.DataFrame(data=x_train_graph,columns=['principle component 1','principle component 2'])

# graphDF=pd.concat([principleDF,y_train])

### LDA

# Instantiate the LDA model and fit it to the train data
lda = LDA(n_components=7)
lda.fit(x_train, y_train)

# Transform the x train data with the lda and output the size
x_train_lda = lda.transform(x_train)
print("Post-LDA shape of x-train:", x_train_lda.shape)

### Univariate Feature Selection

# Create new dataframe with only independent variables by dropping target labels and ID
drop_labels = list(label_cols)
drop_labels.append('ID')
X = data.drop(drop_labels, axis=1).iloc[:,:]


# Create dictionary to store feature results from univariate feature selection
best_features = {}

# Iterate through each target class label
for label in label_cols:

    # Create new dataframe with dependent variables
    y_univariate = data[[label]]

    # Find the best features for classification using chi2 function and select 20 features
    features_new = SelectKBest(score_func=chi2, k=20)
    fit = features_new.fit(X, y_univariate)

    # Obtain scores from the chi2 for each feature
    scores = pd.DataFrame(fit.scores_)

    # Get column names from independent variable dataframe
    features_columns = pd.DataFrame(X.columns)

    # Concatenate dataframes for better formatting
    features_scores = pd.concat([features_columns, scores], axis=1)

    # Add column names
    features_scores.columns = ['Feature', 'Score']

    # Sort dataframe to have highest scores at the top
    features_scores.sort_values(by=['Score'], inplace=True, ascending=False)

    # Uncomment below to see scores of each feature for each target class label
    # print(features_scores.head)

    # Create a sub-dataframe for the top 20 features
    bf = pd.DataFrame(features_scores.iloc[:20,0])

    # Iterate through the top 20 features to count its recurrence within each target class label
    for index, rows in bf.iterrows():
        if rows.Feature not in best_features:
            best_features[rows.Feature] = 1
        else:
            best_features[rows.Feature] += 1

# Print the results of the most important features according to univariate selection
print('Univariate Selection Results: ')
print(sorted(best_features.keys(), reverse=True))
print('Number of features selected: ', len(best_features.keys()))

# # Recursive Feature Elimination
#
# # Iterate
# for label in label_cols:
#     Y = data[label]
#     estimator = SVR(kernel="linear")
#     selector = RFE(estimator, n_features_to_select=20, step=1)
#     selector = selector.fit(X, Y)
#     print('Selected Features: %s' % (selector.support_))
#     print('Feature Ranking: %s' % (selector.ranking_))
