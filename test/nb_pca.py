# Fady Gouda, Abhi Jha, Griffin Noe, Utkrist P Thapa
# CSCI 297-a
# 10/20/20
# Final Project
# nb_pca.py

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
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import *
from sklearn.naive_bayes import *

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
features = ['SR', 'NR', 'OR', 'RR', 'BR', 'MV_A1', 'MV_S52', 'VR_0', 'VR_1', 'VR_2', 'VR_3', 'VR_4',
       'SUR1_1', 'SUR1_2', 'SUR1_4', 'SUR1_6', 'SUR1_7', 'SUR1_9', 'SUR1_10',
       'SUR1_14', 'SUR2_1', 'SUR2_2', 'SUR2_6', 'SUR2_7', 'SUR2_9', 'SUR2_10',
       'SUR2_11', 'SUR3_1', 'SUR3_2', 'SUR3_5', 'SUR3_6', 'SUR3_7', 'SUR3_9',
       'SUR3_10', 'SUR3_11', 'UR_0', 'UR_1', 'UR_3', 'FR_0', 'FR_1', 'FR_2',
       'FR_3', 'FR_4', 'MR_0', 'MR_1', 'MR_2', 'CR_1', 'CR_2', 'TR_1', 'TR_2',
       'TR_5', 'TR_7', 'TR_11', 'TR_12', 'TR_14', 'TR_15']

# Set the x and y from the data frame
source_x = data[features]
y = data[label_cols]

# standardize the dataset
sc = StandardScaler()
sc.fit(source_x)
x = pd.DataFrame(sc.transform(source_x))

accuracies = []
precisions = []
f1s = []
rocs = []
for i in range(10):
    # Split the data at 80/20 train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    #print(x_test.shape)
    #print(y_test.shape)
    print("Original shape of x-train:", x_train.shape)

    ### PCA

    # After running the for loop, it seems that n_components=3 would provide the highest variance,
    # and any component more than that does not add as much variance
    pca=PCA(n_components=7)
    pca.fit(x_train)
    x_train_pca= pca.transform (x_train)
    x_test_pca = pca.transform(x_test)
    print("Post-PCA shape of x-train:", x_train_pca.shape)

    # implementing multi-output classifier with svm
    multi_clf = MultiOutputClassifier(GaussianNB())
    multi_clf.fit(x_train_pca, y_train)
    y_pred = pd.DataFrame(multi_clf.predict(x_test_pca))

    #print(y_test.head)
    #print(pd.DataFrame(y_pred).head)
    #print("Average Precision Score: ", average_precision_score(y_test, y_pred))

    my_accs = []
    for i in range(7):
        diff = abs(y_test.iloc[:, i].values - y_pred.iloc[:, i].values)
        my_accs.append(1 - (sum(diff)/len(diff)))

    accuracies.append(sum(my_accs)/len(my_accs))
    precisions.append(average_precision_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred, average='macro'))
    rocs.append(roc_auc_score(y_test, y_pred))

print("")
print("NB_PCA Results: ")
print("Average Accuracy: ", sum(accuracies)/len(accuracies))
print("Average Precision Score: ", sum(precisions)/len(precisions))
print("Average ROC_AUC Score: ", sum(rocs)/len(rocs))
print("Average F-1 Score: ", sum(f1s)/len(f1s))
