# Fady Gouda, Abhi Jha, Griffin Noe, Utkrist P Thapa
# CSCI 297-a
# 10/20/20
# Final Project
# svm_lda.py

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
from sklearn.metrics import average_precision_score

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

# Set the x and y from the data frame
source_x = data[features]
y = data[label_cols]

# standardize the dataset
sc = StandardScaler()
sc.fit(source_x)
x = pd.DataFrame(sc.transform(source_x))

# Split the data at 80/20 train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape)
print(y_train.shape)
print("Original shape of x-train:", x_train.shape)

results = []
accs = []
for i in range(7):
    lda = LDA(n_components=1)
    lda.fit(x_train, y_train.iloc[:, i])

    # Transform the x train data with the lda and output the size
    x_train_lda = lda.transform(x_train)
    x_test_lda = lda.transform(x_test)
    #print("Post-LDA shape of x-train:", x_train_lda.shape)

    # implementing multi-output classifier with svm
    svm = SVC(kernel='rbf', C=1.0, gamma=0.1)
    svm.fit(x_train_lda, y_train.iloc[:, i])
    y_pred = svm.predict(x_test_lda)
    accs.append(svm.score(x_test_lda, y_test.iloc[:, i]))
    #print(y_pred.shape)
    results.append(y_pred)
y_pred = pd.DataFrame(results).T
print("Average Precision Score: ", average_precision_score(y_test, y_pred))
print("Accuracies: ", accs)
print("Average Accuracy: ", sum(accs)/len(accs))
