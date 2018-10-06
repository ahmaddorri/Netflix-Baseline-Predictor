
import numpy as np
import pandas as pd

combined_data_1_path = 'D:/data sets/netflix/combined_data_1.txt'

combined_data_1 = pd.read_csv(combined_data_1_path, header = None, names = ['Cust_Id', 'Rating'], usecols = [0, 1])

combined_data_1['Rating'] = combined_data_1['Rating'].astype(float)

#########################################

# combined_data_2_path = 'D:/data sets/netflix/combined_data_2.txt'
# combined_data_3_path = 'D:/data sets/netflix/combined_data_3.txt'
# combined_data_4_path = 'D:/data sets/netflix/combined_data_4.txt'
#
# combined_data_2 = pd.read_csv(combined_data_2_path, header = None, names = ['Cust_Id', 'Rating'], usecols = [0, 1])
# combined_data_3 = pd.read_csv(combined_data_3_path, header = None, names = ['Cust_Id', 'Rating'], usecols = [0, 1])
# combined_data_4 = pd.read_csv(combined_data_4_path, header = None, names = ['Cust_Id', 'Rating'], usecols = [0, 1])
#
#
# combined_data_2['Rating'] = combined_data_2['Rating'].astype(float)
# combined_data_3['Rating'] = combined_data_3['Rating'].astype(float)
# combined_data_4['Rating'] = combined_data_4['Rating'].astype(float)
#
# print('Dataset 2 shape:',combined_data_2.shape)
# print('Dataset 3 shape:',combined_data_3.shape)
# print('Dataset 4 shape:',combined_data_4.shape)

#############################################

# load less data for speed

df = combined_data_1
# df = df.append(combined_data_2)
# df = df.append(combined_data_3)
# df = df.append(combined_data_4)

df.index = np.arange(0,len(df))


############################################

df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

################################################


# remove those Movie ID rows
df = df[pd.notnull(df['Rating'])]

df['Movie_Id'] = movie_np.astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)

import pickle

# with open("../data/netflix_dataframe.pickle", 'wb') as output:
#     pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)

# save small part of data for my GIT
with open("../data/netflix_dataframe.pickle", 'wb') as output:
    pickle.dump(df[:100], output, pickle.HIGHEST_PROTOCOL)

