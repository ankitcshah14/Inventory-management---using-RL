import pandas as pd
import datetime
import re
from progress.bar import Bar
import random
import numpy as np

df = pd.read_csv('Historical Product Demand.csv')
# df.reset_index(drop = True, inplace = True)

df = df.dropna()

# Generate Product Costs
MIN_COST, MAX_COST = 100, 1000
product_costs = dict(zip(df['Product_Code'].unique(), np.random.randint(MIN_COST, MAX_COST, len(df['Product_Code'].unique()))))
product_costs = pd.DataFrame(product_costs.items(), columns = ['Product_Code', 'Cost'])
# Convert Date attribute from str to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Removing brackets from Order Demand and converting to int
demands = df[['Order_Demand']].values.flatten().astype('S100')
demands = np.char.lstrip(np.char.rstrip(demands, ')'), '(')
df[['Order_Demand']] = demands.astype('int')

# Finding the top 100 products
N_PRODUCTS = 100
df_grouped_products = df[['Product_Code', 'Order_Demand']].groupby('Product_Code')
df_grouped_products = df_grouped_products['Order_Demand'].agg([np.sum, np.mean])
df_grouped_products = df_grouped_products.sort_values('sum', ascending = False)
df_grouped_products = df_grouped_products.head(N_PRODUCTS)

# Inner Join
df = pd.merge(df, df_grouped_products, how = 'inner', on = 'Product_Code')

# Sort Data according to dates
df = df.sort_values('Date')

print df.head(10)

# Save Top 100 Products
indices = np.arange(N_PRODUCTS)
df_grouped_products['index'] = pd.Series(np.array(indices[::-1]), index = df_grouped_products.index)
df_grouped_products = pd.merge(product_costs, df_grouped_products, how = 'inner', on = 'Product_Code')
df_grouped_products.to_csv('Top 100.csv')

# Save Processed Data
df.to_csv(r'Processed Data.csv')
