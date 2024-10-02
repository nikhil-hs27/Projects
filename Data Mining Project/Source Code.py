# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:31:32 2023

@author: Nikhil
"""

## Loading Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

## Part I ##


## Loading Dataset
A2_df = pd.read_csv("assg2_car - groupD.csv")

#---#

## Data Preprocessing
A2_df.shape
A2_df.dtypes

# Checking for duplicates (additional step)
A2_df.duplicated().any()
# Returns true, so need to remove them

# Removing duplicates
A2_df = A2_df.drop_duplicates()
A2_df.shape
#Dropped 73 records

# Rechecking for duplicates
A2_df.duplicated().any()

# Checking for missing values
A2_df.isna().sum()
# 4 attributes "Buy", "Maint_costs", "person" and "Boot" came up with missing values

"""For "Person" column there are 554 attributes with missing values, 
which is about 33% of the total records. 
Imputing so much values can create a bias in the data. 
So, it is better to remove the whole column from the analysis."""
A2_df = A2_df.drop("Person", axis = 1)
A2_df.shape

"""For "Buy", "Main_cost" and "Boot" columns 
it is better to remove the corresponding records with empty values,
as such imputing categoricals can result in misrepresentation, 
resulting in bad predictions"""
A2_df = A2_df.dropna()
A2_df.shape
#Dropped 97 more records

# Rechecking for missing values
A2_df.isna().sum()

#Transforming the categoricals
#As Quality column is of interest and class label, transforming it manually
A2_df['Quality'] = A2_df['Quality'].replace({
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3})
A2_df.dtypes
#For rest of the columns
for col in A2_df.columns:
    if A2_df[col].dtype == "object":
        A2_df[col] = A2_df[col].astype("category").cat.codes
A2_df.dtypes

#Exporting Preprocessed Data
A2_df.to_csv('20848206_A2_PD_exported.csv', index = False)

#---#

## NN Classifier

#Defining target(class label) and input variables
features = A2_df.drop("Quality", axis=1)
classLabels = A2_df["Quality"]

#Splitting the data into training and testing sets in ratio 75% to 25%
X_train, X_test, y_train, y_test = train_test_split(features, classLabels,
                                                    random_state=1)

#Defining Cost Matrix
CM = np.array([[0, 1, 8 ,10],
               [1, 0, 4, 6],
               [3, 1, 0, 1],
               [6, 2, 1, 0]])

# Model 1 #

# With 1 layer of 120 neurons, optimizer 'sgd', iterations=200, lr = 0.0001
m1 = MLPClassifier(hidden_layer_sizes = (120),
                   max_iter=200,
                   learning_rate_init=0.001,
                   solver='sgd',
                   random_state=1)

m1.fit(X_train, y_train)

#Primary Evaluation
pred = m1.predict(X_test)
tc_m1 = 0
for i in range(len(pred)):
    tc_m1 = tc_m1 + CM[np.array(y_test)[i],pred[i]]
print("Evaluation results for Model 1")
print("Total Cost of missclassification: ", tc_m1)
"""So the target for fine tuning is to minimize this total cost"""

#Other Evaluation measures
print("Loss value", m1.loss_)
print("Accuracy Score: ", accuracy_score(y_test, pred))
print("Senstivity: ", recall_score(y_test, pred, average='macro'))
print("UAR: ", precision_score(y_test, pred, average='macro'))
print("AUC", roc_auc_score(y_test, m1.predict_proba(X_test), multi_class='ovo'))


# Model 2 #

# FineTuning No. of neurons
"""We will test 4 different settings for Neurons as 30, 60, 180, and 240; and
compare it with m1"""
n_neurons = (30, 60, 180, 240)

for nn in n_neurons:
    m_test = MLPClassifier(hidden_layer_sizes = (nn),
                       max_iter=200,
                       learning_rate_init=0.001,
                       solver='sgd',
                       random_state=1)

    m_test.fit(X_train, y_train)

    pred_test = m_test.predict(X_test)
    tc_test = 0
    for i in range(len(pred_test)):
        tc_test = tc_test + CM[np.array(y_test)[i],pred_test[i]]
    
    if tc_test < tc_m1:
        print("No. of Neurons: ", nn)
        print("total_cost", tc_test)
        
"""Conclusion: Changing the no of neurons to 30, decreases the cost.
So, we can take 30 as the neurons for further fineTuning"""

m2 = MLPClassifier(hidden_layer_sizes = (30),
                   max_iter=200,
                   learning_rate_init=0.001,
                   solver='sgd',
                   random_state=1)

m2.fit(X_train, y_train)

#Primary Evaluation
pred_m2 = m2.predict(X_test)
tc_m2 = 0
for i in range(len(pred_m2)):
    tc_m2 = tc_m2 + CM[np.array(y_test)[i],pred_m2[i]]
print("Evaluation results for Model 2")
print("Total Cost of missclassification: ", tc_m2)

#Other Evaluation measures
print("Loss value", m2.loss_)
print("Accuracy Score: ", accuracy_score(y_test, pred_m2))
print("Senstivity: ", recall_score(y_test, pred_m2, average='macro'))
print("UAR: ", precision_score(y_test, pred_m2, average='macro'))
print("AUC", roc_auc_score(y_test, m2.predict_proba(X_test), multi_class='ovo'))


# Model 3 #

# FineTuning No. of hidden layers on model 2
"""We will test 4 different settings for layers as 
(30, 30), (60, 30), (60, 60) and (120, 60, 30); and
compare it with m2"""
n_layers = [(30, 30), (60, 30), (60, 60), (120, 60, 30)]

for nl in n_layers:
    m_test = MLPClassifier(hidden_layer_sizes = nl,
                       max_iter=200,
                       learning_rate_init=0.001,
                       solver='sgd',
                       random_state=1)

    m_test.fit(X_train, y_train)

    pred_test = m_test.predict(X_test)
    tc_test = 0
    for i in range(len(pred_test)):
        tc_test = tc_test + CM[np.array(y_test)[i],pred_test[i]]
    
    if tc_test <= tc_m2:
        print("Layers: ", nl)
        print("total_cost", tc_test)
        
"""Conclusion: Changing the layers doesn't improve our primary evaluation 
metric of  total cost.
So, we will continue with model 2 for further finetuning """

# For illustrating a model the layer change with setting (60, 30) is taken
m3 = MLPClassifier(hidden_layer_sizes = (60, 30),
                   max_iter=200,
                   learning_rate_init=0.001,
                   solver='sgd',
                   random_state=1)

m3.fit(X_train, y_train)

#Primary Evaluation
pred_m3 = m3.predict(X_test)
tc_m3 = 0
for i in range(len(pred_m3)):
    tc_m3 = tc_m3 + CM[np.array(y_test)[i],pred_m3[i]]
print("Evaluation results for Model 2")
print("Total Cost of missclassification: ", tc_m3)

#Other Evaluation measures
print("Loss value", m3.loss_)
print("Accuracy Score: ", accuracy_score(y_test, pred_m3))
print("Senstivity: ", recall_score(y_test, pred_m3, average='macro'))
print("UAR: ", precision_score(y_test, pred_m3, average='macro'))
print("AUC", roc_auc_score(y_test, m3.predict_proba(X_test), multi_class='ovo'))


# Model 4 #

# FineTuning No. of max iterations on model 2
"""We will test 4 different settings for max iterations as 
200, 400, 600, 800; and
compare it with m2"""
n_it = (200, 400, 600, 800)

for ni in n_it:
    m_test = MLPClassifier(hidden_layer_sizes = (30),
                       max_iter=ni,
                       learning_rate_init=0.001,
                       solver='sgd',
                       random_state=1)

    m_test.fit(X_train, y_train)

    pred_test = m_test.predict(X_test)
    tc_test = 0
    for i in range(len(pred_test)):
        tc_test = tc_test + CM[np.array(y_test)[i],pred_test[i]]
    
    if tc_test <= tc_m2:
        print("max_iterations ", ni)
        print("total_cost", tc_test)
        
"""Conclusion: Changing the max iterations to 800, decreases our 
primary evaluation metric total cost to 210.
So, we will consider settings with 800 maximum iterations for further
finetuning"""

m4 = MLPClassifier(hidden_layer_sizes = (30),
                   max_iter=800,
                   learning_rate_init=0.001,
                   solver='sgd',
                   random_state=1)

m4.fit(X_train, y_train)

#Primary Evaluation
pred_m4 = m4.predict(X_test)
tc_m4 = 0
for i in range(len(pred_m4)):
    tc_m4 = tc_m4 + CM[np.array(y_test)[i],pred_m4[i]]
print("Evaluation results for Model 4")
print("Total Cost of missclassification: ", tc_m4)

#Other Evaluation measures
print("Loss value", m4.loss_)
print("Accuracy Score: ", accuracy_score(y_test, pred_m4))
print("Senstivity: ", recall_score(y_test, pred_m4, average='macro'))
print("UAR: ", precision_score(y_test, pred_m4, average='macro'))
print("AUC", roc_auc_score(y_test, m4.predict_proba(X_test), multi_class='ovo'))


# Model 5#

# FineTuning learning rate on model 4
"""We will test 4 different settings for learning rate as 
0.002, 0.004, 0.006, 0.008; and
compare it with m4 which is the best model till now"""
n_lr = (0.002, 0.004, 0.006, 0.008)

for nlr in n_lr:
    m_test = MLPClassifier(hidden_layer_sizes = (30),
                       max_iter=800,
                       learning_rate_init=nlr,
                       solver='sgd',
                       random_state=1)

    m_test.fit(X_train, y_train)

    pred_test = m_test.predict(X_test)
    tc_test = 0
    for i in range(len(pred_test)):
        tc_test = tc_test + CM[np.array(y_test)[i],pred_test[i]]
    
    if tc_test <= tc_m4:
        print("learning rate: ", nlr)
        print("total_cost", tc_test)
        
"""Conclusion: Changing the learning rate to 0.002, decreases our 
primary evaluation criteria total cost.
So, we will consider settings with learning rate 0.002 
for further fine tunning."""

m5 = MLPClassifier(hidden_layer_sizes = (30),
                   max_iter=800,
                   learning_rate_init=0.002,
                   solver='sgd',
                   random_state=1)

m5.fit(X_train, y_train)

#Primary Evaluation
pred_m5 = m5.predict(X_test)
tc_m5 = 0
for i in range(len(pred_m5)):
    tc_m5 = tc_m5 + CM[np.array(y_test)[i],pred_m5[i]]
print("Evaluation results for Model 4")
print("Total Cost of missclassification: ", tc_m5)

#Other Evaluation measures
print("Loss value", m5.loss_)
print("Accuracy Score: ", accuracy_score(y_test, pred_m5))
print("Senstivity: ", recall_score(y_test, pred_m5, average='macro'))
print("UAR: ", precision_score(y_test, pred_m5, average='macro'))
print("AUC", roc_auc_score(y_test, m5.predict_proba(X_test), multi_class='ovo'))


# Model 6 #

# FineTuning Optimizer on model 4
"""We will test the other optimizer setting of 'adam' and
compare it with m5 which is the best model till now"""
sl = 'adam'

m6 = MLPClassifier(hidden_layer_sizes = (30),
                   max_iter=800,
                   learning_rate_init=0.002,
                   solver=sl,
                   random_state=1)

m6.fit(X_train, y_train)

#Primary Evaluation
pred_m6 = m6.predict(X_test)
tc_m6 = 0
for i in range(len(pred_m6)):
    tc_m6 = tc_m6 + CM[np.array(y_test)[i],pred_m6[i]]
print("Evaluation results for Model 6")
print("Total Cost of missclassification: ", tc_m6)

#Other Evaluation measures
print("Loss value", m6.loss_)
print("Accuracy Score: ", accuracy_score(y_test, pred_m6))
print("Senstivity: ", recall_score(y_test, pred_m6, average='macro'))
print("UAR: ", precision_score(y_test, pred_m6, average='macro'))
print("AUC", roc_auc_score(y_test, m6.predict_proba(X_test), multi_class='ovo'))

"""Conclusion: Changing the optimizer, doesn't decrease our primary 
evaluation metric total cost.
So, we will consider settings of model 5 as the best"""

"""Best Model Conclusion: Model 5
Model Architecture(Setting):
    Layers and neurons: (30)
    Max iterations: 800
    learning rate: 0.002
    Optimizer: 'sgd'
    
The above conclusion is based only on the primary evaluation method of 
total cost.    
"""

# Loss Curve for best Model: Model 5
lossPerIteration = m5.loss_curve_ #model loss curve
plt.figure(dpi=125)
plt.plot(lossPerIteration)
plt.title("Loss curve")
plt.xlabel("Number of iterations")
plt.ylabel("Training Loss")
plt.show()

# Confusion Matrix of best Model: Model 4
print(confusion_matrix(y_test, pred_m5))

#---#


## Part II ##

## Reading Dataset
A2_df2 = pd.read_csv("ionoshpere.csv")
A2_df2.shape
A2_df2.dtypes

#Defining labels and features
features = A2_df2.drop('y',axis=1)
classLabels = A2_df2['y']

#Transforming Labels
def tf(y):
    if y == "g":
        return 0
    return 1

classLabels = classLabels.apply(tf)

#Clustering Data
kmeans_est = KMeans(n_clusters = 2, random_state=0)
k_clusters = kmeans_est.fit(features)
pred_p2 = k_clusters.labels_

#Counting and printing Clustering Results
A2_df2['clusters'] = pred_p2
Out_cluster_0 = pd.DataFrame(data = 
    [[len(A2_df2[(A2_df2['clusters'] == 0) & (A2_df2['y'] == 'g')]),
     len(A2_df2[(A2_df2['clusters'] == 0) & (A2_df2['y'] == 'b')])]],
    index = ['Cluster 0'],
    columns = ['y="g"', 'y="b"'])

Out_cluster_1 = pd.DataFrame(data = 
    [[len(A2_df2[(A2_df2['clusters'] == 1) & (A2_df2['y'] == 'g')]),
     len(A2_df2[(A2_df2['clusters'] == 1) & (A2_df2['y'] == 'b')])]],
    index = ['Cluster 1'],
    columns = ['y="g"', 'y="b"'])

Clustering_results = Out_cluster_0.append(Out_cluster_1)
print("Clustering Results: \n", Clustering_results)

#----#