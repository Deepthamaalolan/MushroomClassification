import numpy as nump  
import pandas as pd 
import matplotlib.pyplot as plot
import seaborn as seaborn
from sklearn.preprocessing import LabelEncoder
from collections import Counter

mushroom_dataset_naive_bayes = pd.read_csv('./mushrooms.csv')

mushroom_dataset_naive_bayes.columns = mushroom_dataset_naive_bayes.columns.str.strip().str.replace('-', '_')

mushroom_dataset_naive_bayes = mushroom_dataset_naive_bayes.rename(columns={"class": "m_class"})

mushroom_dataset_naive_bayes.head()

mushroom_dataset_naive_bayes.describe()

for mushroom_type in mushroom_dataset_naive_bayes:
    mushroom_unique_category =nump.unique(mushroom_dataset_naive_bayes[mushroom_type])
    print('{}:{}'.format(mushroom_type,mushroom_unique_category))

mushroom_dataset_naive_bayes = mushroom_dataset_naive_bayes.drop(['veil_type'], axis=1)

mushroom_data_copy= mushroom_dataset_naive_bayes.copy()

mushroom_features_list =[]
for mushroom in mushroom_dataset_naive_bayes:
    mushroom_features= mushroom
    mushroom_features_list.append(mushroom_features)
mushroom_features_list

mushroom_unique_data =[]
for mushroom in mushroom_dataset_naive_bayes:
    unique_mushroom =nump.unique(mushroom_dataset_naive_bayes[mushroom]).tolist()
    mushroom_unique_data.append(unique_mushroom)
mushroom_unique_data

le=LabelEncoder()
for mushroomCol in mushroom_dataset_naive_bayes.columns:
    mushroom_dataset_naive_bayes[mushroomCol] = le.fit_transform(mushroom_dataset_naive_bayes[mushroomCol])

mushroom_odor =list(mushroom_dataset_naive_bayes['odor'].unique())
for odor in mushroom_odor:
    x= mushroom_dataset_naive_bayes[mushroom_dataset_naive_bayes['odor']== odor]
    print('{}:{}'.format(odor,sum(x['m_class'])))

mushroom_odor_2 =list(mushroom_dataset_naive_bayes['odor'].unique())
for odor in mushroom_odor_2:
    odorGroup = mushroom_dataset_naive_bayes[mushroom_dataset_naive_bayes['odor']== odor]
    print('{}:{}'.format(odor ,odorGroup.groupby('m_class').size()))
    print('\n')

mushroom_dataset_naive_bayes.odor.value_counts()

print("total number of poisonous mushroom",len(mushroom_dataset_naive_bayes[mushroom_dataset_naive_bayes.m_class == 1]))

print("total number of edible mushroom", len(mushroom_dataset_naive_bayes[mushroom_dataset_naive_bayes.m_class == 0]))

mushroom_dataset_naive_bayes.head()

mushroom_encoded_list=[]
for mushroom_type in mushroom_dataset_naive_bayes:
    unique_mushroom =nump.unique(mushroom_dataset_naive_bayes[mushroom_type]).tolist()
    mushroom_encoded_list.append(unique_mushroom)
mushroom_encoded_list

for mushroomCol in mushroom_dataset_naive_bayes:
    print('{}:{}'.format(mushroomCol,nump.unique(mushroom_dataset_naive_bayes[mushroomCol])))
print("\n")
for mushroomCol in mushroom_data_copy:
    print('{}:{}'.format(mushroomCol,nump.unique(mushroom_data_copy[mushroomCol])))

mushroom_dataset_naive_bayes.describe()

mushroom_data_yaxis = mushroom_dataset_naive_bayes['m_class'].values
mushroom_data_xaxis = mushroom_dataset_naive_bayes.drop(["m_class"],axis=1)

mushroom_dataset_x = (mushroom_data_xaxis - nump.min(mushroom_data_xaxis))/(nump.max(mushroom_data_xaxis)-nump.min(mushroom_data_xaxis)).values

mushroom_dataset_x

from sklearn.model_selection import train_test_split
mushroom_data_xaxis_train, mushroom_data_xaxis_test, mushroom_data_yaxis_train, mushroom_data_yaxis_test = train_test_split(mushroom_dataset_x,mushroom_data_yaxis,test_size = 0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
mushroom_decision_tree = DecisionTreeClassifier()
mushroom_decision_tree.fit(mushroom_data_xaxis_test,mushroom_data_yaxis_test)
print("Accuracy of decision tree algorithm for test data: ",mushroom_decision_tree.score(mushroom_data_xaxis_test,mushroom_data_yaxis_test))

mushroom_decision_tree.fit(mushroom_data_xaxis_train,mushroom_data_yaxis_train)
print("Accuracy of decision tree algorithm for training data: ", mushroom_decision_tree.score(mushroom_data_xaxis_train,mushroom_data_yaxis_train))

from sklearn.metrics import confusion_matrix
mushroom_y_prediction_decision_tree = mushroom_decision_tree.predict(mushroom_data_xaxis_test)
mushroom_y_true_decision_tree = mushroom_data_yaxis_test
mushroom_confusion_decision_tree = confusion_matrix(mushroom_y_true_decision_tree,mushroom_y_prediction_decision_tree)
fig, axis = plot.subplots(figsize =(5,5))
seaborn.heatmap(mushroom_confusion_decision_tree,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=axis)
plot.xlabel("yaxis_pred")
plot.ylabel("yaxis_true")
plot.show()

from sklearn.model_selection import cross_val_score
mushroom_decision_tree_accuracies = cross_val_score(mushroom_decision_tree,mushroom_dataset_x, mushroom_data_yaxis, cv = 10)
print("Cross validation accuracy scores for mushroom dataset \n", mushroom_decision_tree_accuracies)
print("Average accuracy using 10 cross validation for decision tree: ",nump.mean(mushroom_decision_tree_accuracies))
print("Average std using 10 cross validation for decision tree: ",nump.std(mushroom_decision_tree_accuracies))

from sklearn.metrics import classification_report
yaxis_true = [0]
yaxis_pred = [0]
mushroom_target_names = ['class']
print("Classification report for decision tree algorithm \n" ,classification_report(yaxis_true, yaxis_pred, target_names=mushroom_target_names))

