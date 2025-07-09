import pandas as pd
import pickle


dic = 'thickness_std_d'
distfile = '..\\data\\public\\raw_data\\'
print(distfile + dic + '.pkl', 'rb')
with open(distfile + dic + '.pkl', 'rb') as file:
    data = pickle.load(file)
print(len(data))


grouped_data = {}
df = pd.DataFrame(data.items(), columns=['Index', 'Value'])
df[['Name', 'Part','Part2' ]] = df['Index'].str.split('_', expand=True) #,'Part3'
df = df.sort_values(by=['Name', 'Part'])
df.drop(columns=['Index'], inplace=True)



print(df)


for index, row in df.iterrows():

    name = row['Name']
    value = row['Value']

    if name not in grouped_data:
        grouped_data[name] = []

    grouped_data[name].append(value)



cf = pd.DataFrame(grouped_data)

print(cf)
cf_transposed = cf.transpose()

excel_filename = distfile + dic + '.xlsx'
cf_transposed.to_excel(excel_filename, header=False)

print("Excel ：", excel_filename)





