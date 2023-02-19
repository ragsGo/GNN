import pandas as pd
import numpy as np

usecols = ["Progeny", "Sire", "Dam", "Sex", "G", "NMPrg", "NFPrg", "F", "Homo", "Phen", "Res", "Polygene", "QTL"]
df = pd.read_csv(r'p1_mrk_001.txt', header=None, index_col=0)
ph_df = pd.read_csv(r'p1_data_001.txt')

ph_df.columns = ['1']

cols = np.arange(ph_df['1'].str.split(expand=True).shape[1])
ph_df[cols] = ph_df['1'].str.split(expand=True)
ph_df = ph_df.iloc[:, 1:]

ph_df.columns = usecols
df = df.iloc[1:, :]

df = df.index.values
# print(df[1])
# prints
res = dict()
for i in df:
    # print(i)
    temp = i.split()
    # print(temp)
    # prints
    # for idx, ele in enumerate(temp):
    res[temp[0]] = temp[1]
    # print(res)

# printing result
# print("Dictionary after splits ordering : "
# + str(res))


data_items = res.items()

data_list = list(data_items)

resdf = pd.DataFrame(data_list)
resdf.columns = ['0', '1']
resdf['1'] = resdf['1'].str.replace(r'3|4', '1')

# resdf['1'] =[','.join(ele.split()) for ele in resdf['1'] ]
print(resdf)
for row in resdf.iterrows():
    # print(row[1][1])
    row[1][1] = [','.join(ele.split()) for ele in str(row[1][1])]
    # print(row[1][1])
    # prints

df = pd.DataFrame([pd.Series(x) for x in resdf['1']])
df.columns = ['{}'.format(x + 1) for x in df.columns]
df['id'] = resdf['0']
df['Phen'] = ph_df['Phen']
# print(df)
df.to_csv('mrk.csv', index=False)
