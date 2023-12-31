import pandas as pd
import glob

#xl = pd.ExcelFile('C:/Users/nandh/Desktop/B365 project/nbadraftprediction/collegestats.xlsx')
#for sheet_name in xl.sheet_names:
#    df = xl.parse(sheet_name)
#    df.to_csv(f'{sheet_name}.csv', index=False)

files = glob.glob('C:/Users/nandh/Desktop/B365 project/nbadraftprediction/data/*.csv')
dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

datatokeep = ['NAME','POS','GP','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','REB','AST','Drafted?','Draft Pick','Draft Year']
combined_df = combined_df[datatokeep]
combined_df['Drafted?'] = combined_df['Drafted?'].fillna('No')

print(combined_df)

