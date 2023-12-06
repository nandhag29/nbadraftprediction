import pandas as pd
import glob
import tkinter as tk
from tkinter import messagebox

from webscraper import scrape_player_data
from webscraper import knn
from webscraper import gnb
from webscraper import lr

##################
# DATA PROCESSING
##################

files = glob.glob('./data/*.csv')
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
drafted = data[data['Drafted?'] == 1]
undrafted = data[(data['Drafted?'] == 0) & (data['PTS'] < 10)]
undrafted_sample = undrafted.sample(n=2000, random_state=1)
data = pd.concat([drafted, undrafted_sample], ignore_index=True)
data = data.dropna()

print(data)

############
# BASIC GUI
############

def predict_draft():
    url = url_entry.get()
    stats = scrape_player_data(url)
    prediction = lr(data, stats)
    if prediction[0] == 1:
        messagebox.showinfo("Prediction", "This player will most likely be drafted if they declare for the draft!")
    elif prediction[0] == 0:
        messagebox.showinfo("Prediction", "This player will most likely not be drafted if they declare for the draft!")

root = tk.Tk()
root.title("Draft Prediction")

frame = tk.Frame(root)
frame.pack()

label = tk.Label(frame, text="Enter Sports Reference URL:")
label.pack(side=tk.LEFT)

url_entry = tk.Entry(frame)
url_entry.pack(side=tk.LEFT)

button = tk.Button(frame, text="Predict", command=predict_draft)
button.pack(side=tk.LEFT)

root.mainloop()

