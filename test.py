import pandas as pd
import glob
from analysis import clean_player_data
from analysis import clean_test_data
from analysis import knn
from analysis import gnb
from analysis import lr

files = glob.glob('./data/training/*.csv')
dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data = clean_player_data(data)

testdata = pd.read_csv('./data/test/2022-2023 stats.csv')
testdata = clean_test_data(testdata)

print(data)
print(testdata)


def knntest(data, new_data):
    numcorrect = 0
    for index, row in new_data.iterrows():
        draftstatus = row['Drafted?']
        row_df = pd.DataFrame([row]) 
        classifier = knn(data, row_df)
        if classifier == draftstatus:
            numcorrect += 1
        #print(f"Classifier: {classifier}, Draft Status: {draftstatus}")
    return numcorrect

def gnbtest(data, new_data):
    numcorrect = 0
    for index, row in new_data.iterrows():
        draftstatus = row['Drafted?']
        row_df = pd.DataFrame([row]) 
        classifier = gnb(data, row_df)
        if classifier == draftstatus:
            numcorrect += 1
        #print(f"Classifier: {classifier}, Draft Status: {draftstatus}")
    return numcorrect

def lrtest(data, new_data):
    numcorrect = 0
    for index, row in new_data.iterrows():
        draftstatus = row['Drafted?']
        row_df = pd.DataFrame([row])  
        classifier = lr(data, row_df)
        if classifier == draftstatus:
            numcorrect += 1
        #print(f"Classifier: {classifier}, Draft Status: {draftstatus}")
    return numcorrect


print('KNN Accuracy:')
print(knntest(data, testdata))
print('GNB Accuracy:')
print(gnbtest(data, testdata))
print('LR Accuracy:')
print(lrtest(data, testdata))