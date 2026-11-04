import pandas as pd

df = pd.read_csv("dataset/Drugs/InChiKeys/all_drugs_inchikeys.csv", header=None)

# divide into 3 files
df1 = df.iloc[:len(df)//3]
df2 = df.iloc[len(df)//3:2*len(df)//3]
df3 = df.iloc[2*len(df)//3:]

df1.to_csv("dataset/Drugs/InChiKeys/all_part1.csv", header=None, index=False)
df2.to_csv("dataset/Drugs/InChiKeys/all_part2.csv", header=None, index=False)
df3.to_csv("dataset/Drugs/InChiKeys/all_part3.csv", header=None, index=False)

print(df.shape[0])
print(df1.shape[0])
print(df2.shape[0]) 
print(df3.shape[0])