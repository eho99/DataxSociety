import reflex as rx
import sqlmodel
import ast

import pandas as pd
from sqlalchemy import create_engine

# df = pd.read_csv('mnist_shortlist.csv')
# df['creator_id'] = 1

# def func(x):
#     listOfFloat = []
#     for num in x[1:-1].split(', '):
#         listOfFloat.append(float(num.strip("'")))
#     return listOfFloat

# df['pixel_vals'] = df['pixel_vals'].apply(lambda x: func(x))

# for i in range(500):
#     # df['pixel_vals'][i] = df['pixel_vals'][i]
#     listOfFloat = []
#     for num in df['pixel_vals'][i][1:-1].split(', '):
#         listOfFloat.append(float(num.strip("'")))
#     df['pixel_vals'][i] = listOfFloat
    
# print(type(df['pixel_vals'][0]))

# df.to_csv('mnist_processed.csv', index=False)

new_df = pd.read_csv('mnist_processed.csv')

for i in range(500):
    new_df['id'][i] = i

print(new_df)

engine = create_engine('sqlite:///reflex.db')

table_name = 'mnistdata'
new_df.to_sql(table_name, con=engine, if_exists='replace', index=False)

engine.dispose()
