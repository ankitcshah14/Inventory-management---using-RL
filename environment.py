import pandas as pd
import numpy as np
from prettytable import PrettyTable
import datetime

# Warehouse mapping
# Whse_A = 0
# Whse_C = 1
# Whse_J = 2
# Whse_S = 3

'''
Metric for evaluating convergence ???
'''

class InventoryEnv:
	def __init__(self):
		
		self.df = pd.read_csv('Processed Data.csv')
		# Convert Date attribute from str to datetime
		self.df['Date'] = pd.to_datetime(self.df['Date'])

		self.product_code_index_mapping = pd.read_csv('Top 100.csv')
		self.date = datetime.datetime.strptime('2011-06-24', '%Y-%m-%d')
		self.INV_L, self.INV_H = 0, 5000
		self.N_WHOUSES, self.N_PRODUCTS = 4, 100
		self.observation = np.zeros((self.N_WHOUSES, self.N_PRODUCTS))
		self.action_q = []
		self.CH = 1
		self.action_space_low, self.action_space_high = 100, 10000
		self.LEN_MONTH = 30
		self.whse_mapping = {'Whse_A':0, 'Whse_C':1,'Whse_J':2,'Whse_S':3}

	def reset(self):
		self.observation = np.zeros((self.N_WHOUSES, self.N_PRODUCTS))
		return self.observation

	def action_sample(self):
		return np.random.randint(self.action_space_low, self.action_space_high, (self.N_WHOUSES, self.N_PRODUCTS))

	def order_demand_deduction(self):
		available = np.zeros((self.N_WHOUSES, self.N_PRODUCTS))
		if len(self.action_q) >= self.LEN_MONTH:
			available = self.observation + self.action_q.pop(0)
		else:
			available = self.observation

		# Create Demand Matrix for Current Day
		mask = self.df['Date'] == self.date
		demands_for_current_day = self.df.loc[mask]
		demand_matrix = np.zeros((self.N_WHOUSES, self.N_PRODUCTS))

		for i in demands_for_current_day.index:
			# Extracting the Warehouse index
			whse_num = self.whse_mapping[demands_for_current_day.loc[i, 'Warehouse']]
			
			# Extracting the Product index
			mask = self.product_code_index_mapping['Product_Code'] == demands_for_current_day.loc[i, 'Product_Code']
			prod_num = self.product_code_index_mapping.loc[mask].index[0]

			# Updating the Demand Matrix
			demand_matrix[whse_num][prod_num] += demands_for_current_day.loc[i, 'Order_Demand']

		backorder_cost = 0
		for whse_num in range(self.N_WHOUSES):
			for prod_num in range(self.N_PRODUCTS):
				if available[whse_num][prod_num] >= demand_matrix[whse_num][prod_num]:
					available[whse_num][prod_num] -= demand_matrix[whse_num][prod_num]
				else:
					mask = self.product_code_index_mapping['index'] == prod_num
					CP = int(self.product_code_index_mapping.loc[mask].Cost)
					backorder_cost += CP * (demand_matrix[whse_num][prod_num] - available[whse_num][prod_num])

		return available, -backorder_cost

	def holding_cost(self):
		return -self.CH * np.sum(self.observation)

	def render(self):
		t = PrettyTable(['W_House1', 'W_House2', 'W_House3', 'W_House4'])
		temp = self.observation.T
		
		for i in range(self.N_PRODUCTS):
			t.add_row(temp[i])

		print self.date, '*' * 25
		print t

	def step(self, action):
		action = np.array(action)
		available, backorder_cost = self.order_demand_deduction()
		self.action_q.append(action)
		self.observation = available
		reward = backorder_cost + self.holding_cost()
		self.date += datetime.timedelta(days = 1) #Adding 1 day to date

		return self.observation, reward