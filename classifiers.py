import pandas as pd
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from collections import Counter

##################
# DATA PROCESSING
##################

files = glob.glob('C:/Users/nandh/Desktop/B365 project/nbadraftprediction/data/*.csv')
dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

datatokeep = ['NAME','POS','GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST','Drafted?','Draft Pick','Draft Year']
data = data[datatokeep]
data['Drafted?'] = data['Drafted?'].fillna('No')
data['Draft Pick'] = data['Draft Pick'].fillna(0)
data['Draft Year'] = data['Draft Year'].fillna(0)
mapping_dict = {'G': 1, 'F': 2, 'C': 3, 'G-F': 4, 'F-C': 5, 'F-G': 6, 'C-F': 7, 'C-G': 8, 'G-C': 9, 'SG': 10, 'SF': 10}
data['POS'] = data['POS'].map(mapping_dict)
mapping_dict1 = {'Yes': 1, 'No': 2}
data['Drafted?'] = data['Drafted?'].map(mapping_dict1)

data = data.dropna()
print(data)

##############
# CLASSIFIERS
##############

feature_cols = ['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST']
label_col = 'Drafted?'

X = data[feature_cols]
y = data[label_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("KNN feature importances:")
print(knn)
print("GNB feature importances:")
print(gnb)


