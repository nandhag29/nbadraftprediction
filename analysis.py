import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#########################
# PROCESS AND CLEAN DATA
#########################

def clean_player_data(dataframe):
    datatokeep = ['NAME','POS','GP','MIN','PTS','FGM','FGA','FG%',
                  '3PM','3PA','3P%','FTM','FTA','FT%','REB','AST',
                  'Drafted?','Draft Pick','Draft Year']
    data = dataframe[datatokeep]
    data['Drafted?'] = data['Drafted?'].fillna('No')
    data['Draft Pick'] = data['Draft Pick'].fillna(0)
    data['Draft Year'] = data['Draft Year'].fillna(0)
    mapping_dict = {'G': 1, 'F': 2, 'C': 3, 'G-F': 4, 'F-C': 5, 'F-G': 6, 'C-F': 7, 'C-G': 8, 'G-C': 9, 'SG': 10, 'SF': 10}
    data['POS'] = data['POS'].map(mapping_dict)
    mapping_dict1 = {'Yes': 1, 'No': 0}
    data['Drafted?'] = data['Drafted?'].map(mapping_dict1)
    drafted = data[data['Drafted?'] == 1]
    undrafted = data[(data['Drafted?'] == 0) & (data['PTS'] < 10)]
    undrafted_sample = undrafted.sample(n=2000, random_state=1)
    data = pd.concat([drafted, undrafted_sample], ignore_index=True)
    data = data.dropna()
    return data

def clean_test_data(dataframe):

    datatokeep = ['NAME','POS','GP','MIN','PTS','FGM','FGA','FG%',
                  '3PM','3PA','3P%','FTM','FTA','FT%','REB','AST',
                  'Drafted?','Draft Pick','Draft Year']
    
    data = dataframe[datatokeep]
    data['Drafted?'] = data['Drafted?'].fillna('No')
    data['Draft Pick'] = data['Draft Pick'].fillna(0)
    data['Draft Year'] = data['Draft Year'].fillna(0)
    mapping_dict = {'G': 1, 'F': 2, 'C': 3, 'G-F': 4, 'F-C': 5, 'F-G': 6, 'C-F': 7, 'C-G': 8, 'G-C': 9, 'SG': 10, 'SF': 10}
    data['POS'] = data['POS'].map(mapping_dict)
    mapping_dict1 = {'Yes': 1, 'No': 0}
    data['Drafted?'] = data['Drafted?'].map(mapping_dict1)
    data = data.dropna()
    return data

################
# NEW DATAPOINT
################

def scrape_player_data(url):
    tables = pd.read_html(url)
    stats = tables[1]
    stats = stats[['G', 'MP', 'PTS', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TRB', 'AST']]
    stats.columns = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'REB', 'AST']
    stats = stats.iloc[-2]
    stats = stats.fillna(0)
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
    knn = KNeighborsClassifier()
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
