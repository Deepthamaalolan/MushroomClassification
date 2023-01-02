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

color_columns= []
for mushroomColor in mushroom_dataset_naive_bayes.columns:
    if 'color' in mushroomColor:
        color_columns.append(mushroomColor)
mushroom_color = mushroom_dataset_naive_bayes[color_columns]
mushroom_color.head()

mushroom_color_features=[]
for mushroomColor in mushroom_color:
    color_features= mushroomColor
    mushroom_color_features.append(color_features)
mushroom_color_features

fig,axis=plot.subplots(figsize=(8,8))
seaborn.heatmap(mushroom_color.corr(), annot=True, lineWidth=.5, fmt='.2f',cmap="YlGnBu", ax=axis)
plot.show()

mushroom_dataset_naive_bayes.describe()

mushroom_data_yaxis = mushroom_dataset_naive_bayes.m_class.values
mushroom_data_xaxis = mushroom_dataset_naive_bayes.drop(["m_class"],axis=1)

mushroom_dataset_x = (mushroom_data_xaxis - nump.min(mushroom_data_xaxis))/(nump.max(mushroom_data_xaxis)-nump.min(mushroom_data_xaxis)).values

mushroom_dataset_x

from sklearn.model_selection import train_test_split
mushroom_data_xaxis_train, mushroom_data_xaxis_test, mushroom_data_yaxis_train, mushroom_data_yaxis_test = train_test_split(mushroom_dataset_x,mushroom_data_yaxis,test_size = 0.2,random_state=42)

mushroom_data_xaxis_train

from sklearn.naive_bayes import GaussianNB
mushroom_naive_bayes = GaussianNB()
mushroom_naive_bayes.fit(mushroom_data_xaxis_test,mushroom_data_yaxis_test)
print("Accuracy of naive bayes algorithm for training data: ",mushroom_naive_bayes.score(mushroom_data_xaxis_test,mushroom_data_yaxis_test))

from sklearn.naive_bayes import GaussianNB
mushroom_naive_bayes = GaussianNB()
mushroom_naive_bayes.fit(mushroom_data_xaxis_train,mushroom_data_yaxis_train)
print("Accuracy of naive bayes algorithm for training data: ",mushroom_naive_bayes.score(mushroom_data_xaxis_train,mushroom_data_yaxis_train))

from sklearn.metrics import confusion_matrix

mushroom_y_prediction_naive_bayes = mushroom_naive_bayes.predict(mushroom_data_xaxis_test)
mushroom_y_true_naive_bayes = mushroom_data_yaxis_test
mushroom_confusion_matrix_naive_bayes = confusion_matrix(mushroom_y_true_naive_bayes,mushroom_y_prediction_naive_bayes)
fig, axis = plot.subplots(figsize =(5,5))
seaborn.heatmap(mushroom_confusion_matrix_naive_bayes,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=axis)
plot.xlabel("yaxis_pred")
plot.ylabel("yaxis_true")
plot.show()

from sklearn.model_selection import cross_val_score
mushroom_naive_bayes_accuracies = cross_val_score(mushroom_naive_bayes,mushroom_data_xaxis,mushroom_data_yaxis, cv = 10)
print("Cross validation accuracy scores for mushroom dataset \n", mushroom_naive_bayes_accuracies)
print("Average accuracy using 10 cross validation for Naive Baye's Algorithm: ",nump.mean(mushroom_naive_bayes_accuracies))
print("Average std using 10 cross validation for Naive Baye's Algorithm: ",nump.std(mushroom_naive_bayes_accuracies))

corr = mushroom_dataset_naive_bayes.corr()

seaborn.heatmap(corr)

from sklearn.metrics import classification_report
yaxis_true = [0]
yaxis_prediction = [0]
mushroom_target_names = ['m_class']
print("Classification report for naive bayes algorithm \n" ,classification_report(yaxis_true, yaxis_prediction, target_names=mushroom_target_names))

