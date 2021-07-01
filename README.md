# 297-final
Final project for cs 297a under Profess Cody Watson

The data used in this project can be found at: https://archive.ics.uci.edu/ml/datasets/Amphibians

# EDA
Upon reading the data from the csv, we first transformed the column names from their longhand form into shorthand form for simplification of later tasks. We then specified the classification columns to mutate later and one-hot encoded all of the categorical variables (see feature key below). In order to deal with the 7-label classification we created a 7-digit binary value to represent every permutation of animals then created a mapping to turn the binary into a single value. 

To standardize the data, we first used a simple Standard Scaler as we have no metric to measure the efficacy of the scaler until we introduce the presence of a classifier. When we  start implementing classifiers, we test out all of the types of scalers and study the effect on the classifier. We then split the data at a standard 80/20 train/test split and output the original size of the x_train.

For our first attempt at feature selection, we implemented a principal component analysis of the dataset to find the number of features that maximizes the variance per component. We ran a loop checking the values from 1-20 for the pca test and printed out the explained variance for each value (commented out for succintness of output). Based on the results of the loop, we decided that 3 components was the value that provided the maximal variance from the pcatest. Using that value, we transformed the training data and output the post-pca size of the data. We also tried using 2 components instead of 3 and created a graph to reprecent the pca dataframe. Because the data is in large part categorical, pca is not the ideal method of dimensionality reduction because scaling has no effect on categorical data. We kept this part in to show work done and progress but will probably not use this final dataframe for our classifiers. 

After attempting PCA, we ran an LDA model on the original data and transformed it to get an alternative dataframe to the PCA data. While we know that LDA can be performed after PCA to regularize and avoid overfitting but because the ideal components was 3 for PCA, it left litle room to run LDA afterwards. We used a filler value of 7 to start with and will go back later to examine what the best value for this parameter is.

Finally, we used univariate feature selection by first dropping the id and target labels then iterating thorough each target class variable and identifying the optimal features based on chi squared. After finding the chi squared value for each feature and ordering the dataframe by the chi squared score, we created a sub-dataframe of the top 20 features. We simply selected 20 as a filler value but will go back later to examine what values may be better. We then iterated through these top 20 features and counted the recurrence within each target class label. Finally, we printed the most important features from the univariate selection as well as the number of features selected.
# KNN
We implemented KNeighborsClassifier imported from sklearn.neighbors on two different approaches for feature extraction: LDA and PCA.

### KNN_LDA:
File 297-final/test/knn_lda.py contains our implementation of LDA for feature extraction in a KNN. After the initial data preparation and feature scaling, we split the data into 80% training and 20% testing sets. Because this is a multilabel classification task, we decided to run the LDA and the classifier on each of the seven target labels individually, and take the macro average of the performance metric to gauge the performance. We use LinearDiscriminantAnalysis imported from sklearn.discriminant_analysis to reduce 54 features to 1 component. This is because the number of components in the LDA is given by min(number of classes - 1, number of features). Since we have a separate LDA and classifier for each of the target labels, the number of classes becomes binary (presence/absence of a certain type of frog) and n_components = 2 - 1 = 1. We calculate accuracy, precision, f1-score and roc_auc_score for each label, and take the macro average. We run this entire process 10 times, and get an average of each performance metric across ten iterations. 
We started off with default parameters for the KNN. We varied n_neighbors from 3 to 10 which did not seem to bring much improvement in performance. It was the same with the distance metric. 

### KNN_PCA:
File 297-final/test/knn_pca.py contains our implementation of PCA for feature extraction in a KNN. The feature extraction is implemented the same way as the LDA, except that the number of components used here is 7. This is because we found that the majority of the variance in the dataset can be encompassed with 7 components. The classifier we used is MultiOutputClassifier imported from sklearn.multioutput. We calculate the same performance metrics in the same way as mentioned in kdd_lda, and we find the macro average for all metrics. Then we find the average of these metrics over 10 iterations, and report the numbers in the table in our short research paper. 
We varied the hyperparameters the same way as mentioned above in knn_lda, and found that it did not improve the performance metric to standards comparable to some of the other better performing models.

# Naive Bayes
We implemented GaussianNB imported from sklearn.naive_bayes on two different approaches for feature extraction: LDA and PCA.

### NB_LDA:
File 297-final/test/nb_lda.py contains our implementation of LDA for feature extraction in a GaussianNB. After the initial data preparation and feature scaling, we split the data into 80% training and 20% testing sets. We use LinearDiscriminantAnalysis imported from sklearn.discriminant_analysis to reduce 54 features to 1 component. This is because the number of components in the LDA is given by min(number of classes - 1, number of features). Since we have a separate LDA and classifier for each of the target labels, the number of classes becomes binary (presence/absence of a certain type of frog) and n_components = 2 - 1 = 1. We calculate accuracy, precision, f1-score and roc_auc_score for each label, and take the macro average. We run this entire process 10 times, and get an average of each performance metric across ten iterations. 
We explored a variety of choices for naive bayes classifiers including BernoulliNB. We found that the model performed best on unseen test data in terms of accuracy and precision with GaussianNB as the classifier.

### NB_PCA:
File 297-final/test/nb_pca.py contains our implementation of PCA for feature extraction in a GaussianNB. The feature extraction is implemented the same way as the LDA, except that the number of components used here is 7. This is because we found that the majority of the variance in the dataset can be encompassed with 7 components. The classifier we used is MultiOutputClassifier imported from sklearn.multioutput. We calculate the same performance metrics in the same way as mentioned in nb_lda, and we find the macro average for all metrics. Then we find the average of these metrics over 10 iterations, and report the numbers in the table in our short research paper. 
We varied the classifiers the same way as mentioned above in nb_lda, and found that GaussianNB performs best in terms of accuracy and precision.

# SVM

File 297-final/test/svm_uni_pca_lda.py contains the implementation of multiple data techniques with a support vector machine classifier. First, we apply the univariate feature selection strategy to find the features that are most relevant to the target values. Scores for each individual features are calculated on the basis of the chi-squared function. We choose twenty features with the highest chi-squared scores. We then use the feature extraction methods. For labels 4 and 5, we apply LDA, and PCA for rest of the features. Finally, we use the support vector classifier with a linear kernel, C value of 0.2 (since the degree of regularization is inversely proportional to the magnitude), and a gamma value of 0.05. We the run the model 10 times to calculate average scores of accuracy, ROC, precision, and F1. These numbers have been reported in the table in the paper. 

# Random Forest
We implemented RandomForestClassifier imported from sklearn.ensemble on two different approaches for feature extraction: LDA and PCA.

### RF_LDA:
File 297-final/test/rf_lda.py contains our implementation of LDA for feature extraction in a random forest. We use LinearDiscriminantAnalysis imported from sklearn.discriminant_analysis to reduce 54 features to 1 component because the number of components in the LDA is given by min(number of classes - 1, number of features). We calculate accuracy, precision, f1-score and roc_auc_score for each label, and take the macro average. We run this entire process 10 times, and get an average of each performance metric across ten iterations. 
We started off with default parameters for the random forest. We varied n_estimators from 80 to 300 which did not bring significant improvement in performance. 

### RF_PCA:
File 297-final/test/rf_pca.py contains our implementation of PCA for feature extraction in a random forest. The feature extraction is implemented the same way as the LDA, except that the number of components used here is 7. This is because we found that the majority of the variance in the dataset can be encompassed with 7 components. The classifier we used is MultiOutputClassifier imported from sklearn.multioutput. We calculate the same performance metrics in the same way as mentioned in knn_lda, and we find the macro average for all metrics. Then we find the average of these metrics over 10 iterations, and report the numbers in the table in our short research paper. 
We varied the hyperparameters the same way as mentioned above in rf_lda, and found that it did not significantly improve performance.

# Logistic Regression
We implemented LogisticRegression imported from sklearn.linear_model on threee approaches for feature extraction: LDA, PCA, and Univariate. 

### LR_LDA:
File 297-final/test/lr_lda.py contains our implemention of LDA feature extraction with a logistic regression algorithm. As previously mentioned, the number of components must be equal to 1 for LDA when the problem has a binary output (even if it is multilabel). For the model itself, we used a grid search followed by manual tweaking of the c parameter to hone in on 0.6 as the ideal value. We attempted to run a penalty of both l2 and none (l1 does not work with the data) and found that l2 works significantly better than none. 

### LR_PCA:
File 297-final/lr_pca.py contains our implementation of PCA feature extraction with a logistic regression algorithm. This model is our 'main' model which is why it is located in the main folder and was selected based on our selected performance metrics. For this logistic regression, we did the same parameter exploration as mentioned in the LDA portion and found that 0.6 was again an ideal value of c. Tweaking to values such as 0.3-0.5 presented potential accuracy improvements but they were so small that we decided to stick with the standard 0.6 value from LDA. For the pca parameters, we ran a for loop from 1 to 50 and found that values from 20-28 seemed to perform the best after a handful of test runs so 23 was selected for final evaluation. 

### LR_Univariate:
File 297-final/test/lr_univariate.py contains our implementation of univariate feature selection with a logistic regression algorithm. For this approach we decided to use the same 20 feature parameter cutoff as the svm univariate implementation for initial use. We then tweaked the values for the feature cutoff betwen 5 and 30 and found that 10 optimized all of the performance metrics so we selected that. We also attempted to use the same feature extraction engineering by layering on selective pca on top of the initial univariate selection and found that it increased the accuracy but decreased the f1 and roc auc scores so we decided to stick with just the univariate selection. 

# Features Key
List of attributes and types:
ID -> Integer
MV -> Categorical
SR -> Numerical
NR -> Numerical
TR -> Categorical
VR -> Categorical
SUR1 -> Categorical
SUR2 -> Categorical
SUR3 -> Categorical
UR -> Categorical
FR -> Categorical
OR -> Numerical
RR -> Ordinal;
BR -> Ordinal;
MR -> Categorical
CR -> Categorical
Green frogs -> Categorical; Label 1
Brown frogs -> Categorical; Label 2
Common toad -> Categorical; Label 3
Fire-bellied toad -> Categorical; Label 4
Tree frog -> Categorical; Label 5
Common newt -> Categorical; Label 6
Great crested newt -> Categorical; Label 7

Name and symbol type description:
1) ID - vector ID (not used in the calculations)
2) MV - motorway (not used in the calculations)
3) SR - Surface of water reservoir numeric 
4) NR - Number of water reservoirs in habitat - Comment: The larger the number of reservoirs, the more likely it is that some of them will be suitable for amphibian breeding.
5) TR - Type of water reservoirs:
    a. reservoirs with natural features that are natural or anthropogenic water reservoirs (e.g., subsidence post-exploited water reservoirs), not subjected to naturalization
    b. recently formed reservoirs, not subjected to naturalization
    c. settling ponds
    d. water reservoirs located near houses
    e. technological water reservoirs
    f. water reservoirs in allotment gardens
    g. trenches
    h. wet meadows, flood plains, marshes
    i. river valleys
    j. streams and very small watercourses
6) VR - Presence of vegetation within the reservoirs:
    a. no vegetation
    b. narrow patches at the edges
    c. areas heavily overgrown
    d. lush vegetation within the reservoir with some part devoid of vegetation
    e. reservoirs completely overgrown with a disappearing water table
    Comment: The vegetation in the reservoir favors amphibians, facilitates breeding, and allows the larvae to feed and give shelter. However, excess vegetation can lead to the overgrowth of the pond and water shortages.
7) SUR1 - Surroundings; the dominant types of land cover surrounding the water reservoir
8) SUR2 - Surroundings; the second most dominant types of land cover surrounding the water reservoir
9) SUR3 - Surroundings; the third most dominant types of land cover surrounding the water reservoir
    Comment: The surroundings feature was designated in three stages. First, the dominant surroundings were selected. Then, two secondary types were chosen.
    a. forest areas (with meadows) and densely wooded areas
    b. areas of wasteland and meadows
    c. allotment gardens
    d. parks and green areas
    e. dense building development, industrial areas
    f. dispersed habitation, orchards, gardens
    g. river valleys
    h. roads, streets
    i. agricultural land
    The most valuable surroundings of water reservoirs for amphibians are areas with the least anthropopressure and proper moisture.
10) UR - Use of water reservoirs:
    a. unused by man (very attractive for amphibians)
    b. recreational and scenic (care work is performed)
    c. used economically (often fish farming)
    d. technological
11) FR - The presence of fishing:
    a. lack of or occasional fishing
    b. intense fishing
    c. breeding reservoirs
    Comment: The presence of a large amount of fishing, in particular predatory and intense fishing, is not conducive to the presence of amphibians.
12) OR - Percentage access from the edges of the reservoir to undeveloped areas (the proposed percentage ranges are a numerical reflection of the phrases: lack of access, low access, medium access, large access to free space):
    a. 0-25; lack of access or poor access
    b. 25-50; low access
    c. 50-75; medium access,
    d. 75-100; large access to terrestrial habitats of the shoreline is in contact with the terrestrial habitat of amphibians.
13) RR Minimum distance from the water reservoir to roads:
    a. <50 m
    b. 50-100 m
    c. 100-200 m
    d. 200-500 m
    e. 500-1000 m
    f. >1000 m
    Comment: The greater the distance between the reservoir and the road, the more safety for amphibians.
14) BR - Building development - Minimum distance to buildings:
    a. <50 m
    b. 50-100 m
    c. 100-200 m
    d. 200-500 m
    e. 500-1000 m
    f. >1000 m
    Comment: The more distant the buildings, the more favorable the conditions for the occurrence of amphibians.
15) MR - Maintenance status of the reservoir:
    a. Clean
    b. slightly littered
    c. reservoirs heavily or very heavily littered
    Comment: Trash causes devastation of the reservoir ecosystem. Backfilling and leveling of water reservoirs with ground and debris should also be considered.
16) CR - Type of shore
    a. Natural
    b. Concrete
    Comment: A concrete shore of a reservoir is not attractive for amphibians. A vertical concrete shore is usually a barrier for amphibians when they try to leave the water.
17) Label 1 - the presence of Green frogs
18) Label 2 - the presence of Brown frogs
19) Label 3 - the presence of Common toad
20) Label 4 - the presence of Fire-bellied toad
21) Label 5 - the presence of Tree frog
22) Label 6 - the presence of Common newt
23) Label 7 - the presence of Great crested newt


