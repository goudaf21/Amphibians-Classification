# Fady Gouda, Abhi Jha, Griffin Noe, Utkrist P Thapa
# CSCI 297-a
# 10/20/20
# Final Project
# lr_univariate.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression

def main():
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
    features = ['ID', 'SR', 'NR', 'OR', 'RR', 'BR', 'MV_A1', 'MV_S52', 'VR_0', 'VR_1', 'VR_2', 'VR_3', 'VR_4',
           'SUR1_1', 'SUR1_2', 'SUR1_4', 'SUR1_6', 'SUR1_7', 'SUR1_9', 'SUR1_10',
           'SUR1_14', 'SUR2_1', 'SUR2_2', 'SUR2_6', 'SUR2_7', 'SUR2_9', 'SUR2_10',
           'SUR2_11', 'SUR3_1', 'SUR3_2', 'SUR3_5', 'SUR3_6', 'SUR3_7', 'SUR3_9',
           'SUR3_10', 'SUR3_11', 'UR_0', 'UR_1', 'UR_3', 'FR_0', 'FR_1', 'FR_2',
           'FR_3', 'FR_4', 'MR_0', 'MR_1', 'MR_2', 'CR_1', 'CR_2', 'TR_1', 'TR_2',
           'TR_5', 'TR_7', 'TR_11', 'TR_12', 'TR_14', 'TR_15']

    collinear_pairs = [['TR_1', 'UR_3'], ['UR_3','UR_0'],['FR_0','UR_0'],['CR_1','CR_2'],['CR_2','MR_1'],['CR_1','MR_1']
                        ,['MR_2','MR_0']]

    drop_labels = list(label_cols)
    drop_labels.append('ID')
    drop_labels.append('MV_A1')
    drop_labels.append('MV_S52')
    X = data.drop(drop_labels, axis=1).iloc[:,:]

    # Create dictionary to store feature results from univariate feature selection
    best_features = {}

    averageTotal = 0
    rocTotal = 0
    f1Total = 0
    precisionTotal = 0

    # Iterate through each target class label
    for label in label_cols:



        # Create new dataframe with dependent variables
        y_univariate = data[label]

        # Find the best features for classification using chi2 function and select 10 features
        features_new = SelectKBest(score_func=f_classif, k=10)
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


        #print(best_features)

    new_features = list(best_features.keys())
    #print(new_features)

    new_features.remove('TR_15')
    new_features.remove('TR_14')
    new_features.remove('TR_12')
    new_features.remove('UR_3')
    new_features.remove('FR_1')
    new_features.remove('FR_2')
    #new_features.remove('FR_4')
    new_features.remove('CR_1')
    new_features.remove('MR_0')


    source_x = data.drop(data.columns.difference(new_features[:16]), axis=1)

    # standardize the dataset
    sc = StandardScaler()
    sc.fit(source_x)
    x = pd.DataFrame(sc.transform(source_x))

    for label in label_cols:

        y = data[label]


        # Split the data at 80/20 train/test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        #print(x_test.shape)
        #print(y_test.shape)
        #print("Original shape of x-train:", x_train.shape)


        # # implementing multi-output classifier with svm
        # multi_clf = MultiOutputClassifier(SVC(kernel="poly", deg=4))
        # multi_clf.fit(x_train, y_train)
        # y_pred = multi_clf.predict(x_test)
        # #print(y_test.head)
        # #print(pd.DataFrame(y_pred).head)
        # print("Average Precision Score: ", average_precision_score(y_test, y_pred))

        # if label=='L3' or label=='L4' or label=='L5':
        #     pca=LDA(n_components=1)
        #     pca.fit(x_train, y_train)
        #     x_train= pca.transform (x_train)
        #     x_test = pca.transform(x_test)
        # else:
        #     pca=PCA(n_components=5)
        #     pca.fit(x_train)
        #     x_train= pca.transform (x_train)
        #     x_test = pca.transform(x_test)
        #     #print("Post-PCA shape of x-train:", x_train_pca.shape)

        lr = LogisticRegression(C=0.6, penalty='l2', random_state=1)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)

        average = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rocTotal += roc
        averageTotal += average
        f1Total += f1
        precisionTotal += prec
        # print("Accuracy: %.2f" % (average))

    # print("Average accuracy: %.3f" % (averageTotal/7))
    # print("")
    # print("Average auc: %.3f" % (rocTotal/7))
    # print("")
    return averageTotal/7, rocTotal/7, f1Total/7, precisionTotal/7

if __name__ == "__main__":
    average = 0
    roc = 0
    f1 = 0
    prec = 0
    for x in range(10):
        average += main()[0]
        roc += main()[1]
        f1 += main()[2]
        prec += main()[3]

    print("__________________________________")
    print("")
    print("Accuracy Average: %.3f" % ((average/10)*100))
    print("ROC Average: %.3f" % ((roc/10)*100))
    print("F1 Average: %.3f" % ((f1/10)*100))
    print("Precision Average: %.3f" % ((prec/10)*100))
