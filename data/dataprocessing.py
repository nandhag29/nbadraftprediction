import pandas as pd

xl = pd.ExcelFile('C:/Users/nandh/Desktop/B365 project/nbadraftprediction/collegestats.xlsx')

for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)
    df.to_csv(f'{sheet_name}.csv', index=False)
