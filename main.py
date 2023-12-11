import pandas as pd
import glob
import tkinter as tk
from tkinter import messagebox

from analysis import clean_player_data
from analysis import scrape_player_data
from analysis import knn
from analysis import gnb
from analysis import lr

##################
# DATA PROCESSING
##################

files = glob.glob('./data/training/*.csv')
dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

data = clean_player_data(data)

############
# BASIC GUI
############

def predict_draft():
    url = url_entry.get()
    stats = scrape_player_data(url)
    print(stats)
    prediction = lr(data, stats)
    print('KNN Prediction:')
    print(knn(data, stats))
    print('GNB Prediction:')
    print(gnb(data, stats))
    print('LR Prediction:')
    print(lr(data, stats))
    if prediction[0] == 1:
        messagebox.showinfo("Prediction", "This player will most likely be drafted if they declare for the draft!")
    elif prediction[0] == 0:
        messagebox.showinfo("Prediction", "This player will most likely not be drafted if they declare for the draft.")

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

