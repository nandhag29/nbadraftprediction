import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

################
# NEW DATAPOINT
################

def scrape_player_data(url):
    tables = pd.read_html(url)
    stats = tables[1]
    stats = stats[['G', 'MP', 'PTS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TRB', 'AST']]
    stats.columns = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'REB', 'AST']
    stats = stats.iloc[-2]
    stats = stats.to_frame().T
    return stats

##############
# CLASSIFIERS
##############

feature_cols = ['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST']
label_col = 'Drafted?'

#K-Nearest Neighbors
def knn(data, new_data):
    feature_cols = ['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST']
    label_col = 'Drafted?'
    knn = KNeighborsClassifier(n_neighbors=14)
    X = data[feature_cols]
    y = data[label_col]
    knn.fit(X, y)
    new_data = new_data[feature_cols]
    prediction = knn.predict(new_data)
    return prediction

#Naive Bayes
def gnb(data, new_data):
    feature_cols = ['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST']
    label_col = 'Drafted?'
    gnb = GaussianNB()
    X = data[feature_cols]
    y = data[label_col]
    gnb.fit(X, y)
    new_data = new_data[feature_cols]
    prediction = gnb.predict(new_data)
    return prediction

#Logistic Regression 
def lr(data, new_data):
    feature_cols = ['GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST']
    label_col = 'Drafted?'
    lr = LogisticRegression(max_iter = 20000)
    X = data[feature_cols]
    y = data[label_col]
    lr.fit(X, y)
    new_data = new_data[feature_cols]
    prediction = lr.predict(new_data)
    return prediction
