import pandas as pd
import glob
import json
import os


print(os.getcwd())
files = glob.glob('tweet/*')
length = len(files)

print(length)


dictlist = []

for file in files:
    json_string = open(file, 'r', encoding="utf-8").read()
    json_dict = json.loads(json_string)
    dictlist.append(json_dict)

df = pd.DataFrame(dictlist)
print(df)

df = df.replace({'\n': ' '}, regex=True)
df = df.replace({'\t': ' '}, regex=True)
df = df.replace({'\r': ' '}, regex=True)

df
print(df)

df.to_csv("data.csv")

print('Completed writing to csv')


