import pandas as pd
# from scipy.stats import zscore
#
# polyFrame = pd.read_csv('D:\\dataPoly\\Белоус Полина.csv')
#
# df = polyFrame[polyFrame['trial'].notnull()]
# dfTest = polyFrame[polyFrame['trial'].notnull()]
#
# # t1 = df[df['trial'] == 1.0]
# # score = zscore(t1['Neutral'])
#
# # new[new['trial'] == 1.0]['Neutral'] = zscore(new[new['trial'] == 1.0]['Neutral'])
#
# for trialNum in 1, 2, 3:
#     for col in range(5, df.shape[1]):
#         df.loc[df['trial'] == trialNum, df.columns[col]] = zscore(df.loc[df['trial'] == trialNum, df.columns[col]])

df = pd.DataFrame({'c1': ['bear'],
                  'c2': [1864],
                   'c3': [22000],
                   'c4': ['alo']})

i = 0
j = 0
print(df.shape[1])
for key, value in df.items():
    if i < df.shape[1]-1:
        j += 1
    else:
        j = i
    print(f'label: {key}')
    print(f'content: {value}', sep='\n')
    print(df.iloc[:, j])
    i += 1



