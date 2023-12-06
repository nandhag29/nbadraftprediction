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
mapping_dict = {'G': 1, 'F': 2, 'C': 3, 'G-F': 4, 'F-C': 5, 'F-G': 6, 'C-F': 7, 'C-G': 8, 'G-C': 9, 'SG': 10, 'SF': 10}
data['POS'] = data['POS'].map(mapping_dict)
mapping_dict1 = {'Yes': 1, 'No': 2}
data['Drafted?'] = data['Drafted?'].map(mapping_dict1)
print(data)

##############
# CLASSIFIERS
##############

feature_cols = ['POS','GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST']
label_col = 'Drafted?'

X = data[feature_cols]
y = data[label_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("KNN feature importances:")
print(knn.feature_importances_)
print("GNB feature importances:")
print(gnb.feature_importances_)


# hardcoded classifiers
class hardcodedKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

class hardcodedNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = []
        for i, c in enumerate(self.classes):
            X_c = X[np.where(y == c)]
            self.parameters.append([])
            for col in X_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

    def _pdf(self, class_idx, x):
        output = 1
        parameters = self.parameters[class_idx]
        for i in range(len(parameters)):
            mean = parameters[i]["mean"]
            var = parameters[i]["var"]
            numerator = np.exp(-(x[i]-mean)**2 / (2 * var))
            denominator = np.sqrt(2 * np.pi * var)
            output *= numerator / denominator
        return output

    def _predict(self, x):
        posteriors = []

        for i, c in enumerate(self.classes):
            prior = np.log(1 / len(self.classes))
            posterior = np.sum(np.log(self._pdf(i, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)