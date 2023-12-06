import pandas as pd
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
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
mapping_dict1 = {'Yes': 1, 'No': 0}
data['Drafted?'] = data['Drafted?'].map(mapping_dict1)

data = data.dropna()
print(data)

################
# NEW DATAPOINT
################

def scrape_sports_reference_player_stats(url):
    tables = pd.read_html(url)
    totals_table = tables[1]
    return totals_table

url = 'https://www.sports-reference.com/cbb/players/brandon-miller-3.html'
stats = scrape_sports_reference_player_stats(url)
stats = stats[['G', 'MP', 'PTS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TRB', 'AST']]
stats.columns = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'REB', 'AST']
stats = stats.iloc[-2]
stats = stats.to_frame().T

print(stats)


##############
# CLASSIFIERS
##############

feature_cols = ['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST']
label_col = 'Drafted?'

#K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=14)

X = data[feature_cols]
y = data[label_col]
knn.fit(X, y)

new_data = stats[feature_cols]

prediction = knn.predict(new_data)

print(prediction)

#Naive Bayes
gnb = GaussianNB()

gnb.fit(X, y)

predictions = gnb.predict(new_data)

print(predictions)

#Logistic Regression 
lr = LogisticRegression()

lr.fit(X, y)

new_data = stats[feature_cols]

predictions1 = lr.predict(new_data)

print(predictions1)