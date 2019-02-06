import pandas as pd
import numpy as np
from prettytable import PrettyTable
import datetime

#PREVIOUS WORKING VER : environment_cost_compute.py
# Warehouse mapping
# Whse_A = 0
# Whse_C = 1
# Whse_J = 2
# Whse_S = 3

'''
Metric for evaluating convergence ???
'''


'''
TODO:
	DONE INCORPORATE REWARDS FOR SELLING STUFF IN ENV.STEP
	DONE ENFORCE limit for stock size of inventory using self.INV_H, self.INV_L

	DONE CLIP NEGATIVE ACTIONS TO ZERO
	DONE MOVE PRINTS OF DIFFERENT COSTS FROM STEP() TO RENDER() FUNCTION
	DONE ADD DATE TO THE STATE
	DONE ADD SOME VARIANCE TO DEMAND FOR EACH PRODUCT AT EACH WAREHOUSE
	VECTORISE ORDER DEDUCTIONS
	DONE VECTORISE COST COMPUTATION
	CURRENTLY ONLY DEALING WITH ONE PRODUCT : SCALE TO ENTIRE 4 x 100 (LINE 102)

'''

class Space:
	def __init__(self, shape_row, shape_col, low, high):
		self.shape = [shape_row, shape_col]
		self.high = high
		self.low = low

	def sample(self):
		return np.random.randint(self.low, self.high, (self.shape[0], self.shape[1]))

class InventoryEnv:

	def init_env(self):

		self.date = datetime.datetime.strptime('2011-06-24', '%Y-%m-%d')
		self.observation = np.zeros((self.N_WHOUSES, self.N_PRODUCTS))
		self.available = np.zeros((self.N_WHOUSES, self.N_PRODUCTS))
		self.action_q = []
		self.order_cost = 0
		self.holding_cost = 0
		self.backorder_cost = 0
		self.revenue_gen = 0
		self.day_of_year = 1
		self.obs_flat = self.observation.flatten()
		self.obs_flat_date = np.hstack([self.obs_flat,np.array([self.day_of_year])])
		self.obs_flat_date = np.reshape(self.obs_flat_date, (self.obs_flat_date.shape[0],1))



	def __init__(self):
		
		self.df = pd.read_csv('Processed Data.csv')
		# Convert Date attribute from str to datetime
		self.df['Date'] = pd.to_datetime(self.df['Date'])
		self.product_code_index_mapping = pd.read_csv('Top 100.csv')
		self.INV_L, self.INV_H = 0, 90000000
		self.N_WHOUSES, self.N_PRODUCTS = 4, 1
		self.action_space_low, self.action_space_high = 0, 90000000

		self.observation_space = Space(self.N_WHOUSES * self.N_PRODUCTS + 1, 1, self.INV_L, self.INV_H)
		self.action_space = Space(self.N_WHOUSES, self.N_PRODUCTS, self.action_space_low, self.action_space_high)

		self.init_env()

		self.PROFIT_FACTOR = 0.5
		self.CP_VECTOR = np.zeros((self.N_PRODUCTS,1))
		for prod_num in range(self.N_PRODUCTS):
			mask = self.product_code_index_mapping['index'] == prod_num
			self.CP_VECTOR[prod_num] = int(self.product_code_index_mapping.loc[mask].Cost)

		self.SP_VECTOR = (1. + self.PROFIT_FACTOR) * self.CP_VECTOR

		self.CH = 1
		self.LEN_MONTH = 30
		self.DEMAND_VARIANCE_FACTOR = 0.2
		self.whse_mapping = {'Whse_A':0, 'Whse_C':1,'Whse_J':2,'Whse_S':3}

	def reset(self):
		self.init_env()
		return self.obs_flat_date


	def order_demand_deduction(self):
		if len(self.action_q) >= self.LEN_MONTH:
			self.available = self.observation + self.action_q.pop(0)
			self.available[self.available > self.observation_space.high] = self.observation_space.high
		else:
			self.available = self.observation

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
			# if prod_num < self.N_PRODUCTS:
			demand_temp = np.int((1. + np.random.uniform(-self.DEMAND_VARIANCE_FACTOR, self.DEMAND_VARIANCE_FACTOR)) * demands_for_current_day.loc[i, 'Order_Demand'])
			demand_matrix[whse_num][0] += demand_temp

		self.backorder_cost = 0
		self.revenue_gen = 0
		
		# print '*' * 50
		# print self.available
		# print '%' * 50
		# print demand_matrix


		temp_demand_s = demand_matrix.copy() #to store those item demands that were fulfilled
		temp_demand_b = demand_matrix.copy() #to store those item demands that were not fulfilled
		#set all those demand mtx elements to zero, that are NOT unfulfilled demands::
		temp_demand_s[self.available < demand_matrix] = 0

		#sale done, updating availability mtx::
		available_temp_s = self.available - temp_demand_s #to store availability after sales

		#set all those demand mtx elements to zero, that ARE fulfilled orders::
		temp_demand_b[self.available >= demand_matrix] = 0

		# print '*' * 50
		# print temp_demand_s
		# print '%' * 50
		# print temp_demand_b

		available_temp_b = self.available.copy()
		available_temp_b[temp_demand_b == 0] = 0 #set all those available mtx elements to zero, that ARE fulfilled orders
		
		backorder_mtx = temp_demand_b - available_temp_b

		#computing back-order costs::
		total_per_product_amounts = backorder_mtx.sum(axis = 0)
		total_per_product_amounts = np.reshape(total_per_product_amounts, (total_per_product_amounts.shape[0], 1))
		self.backorder_cost = np.dot(self.CP_VECTOR.T, total_per_product_amounts)

		#computing revenue::
		total_sold_per_product_amounts = temp_demand_s.sum(axis = 0)
		total_sold_per_product_amounts = np.reshape(total_sold_per_product_amounts, (total_sold_per_product_amounts.shape[0], 1))
		self.revenue_gen = np.dot(self.SP_VECTOR.T, total_sold_per_product_amounts)

		self.available = available_temp_s

		# for whse_num in range(self.N_WHOUSES):
		# 	for prod_num in range(self.N_PRODUCTS):
		# 		if demand_matrix[whse_num][prod_num] > 0:
		# 			if self.available[whse_num][prod_num] >= demand_matrix[whse_num][prod_num]:
		# 				self.available[whse_num][prod_num] -= demand_matrix[whse_num][prod_num]
		# 				self.revenue_gen += SP * demand_matrix[whse_num][prod_num]
		# 			else:
		# 				self.backorder_cost += CP * (demand_matrix[whse_num][prod_num] - self.available[whse_num][prod_num])

		return self.available, -self.backorder_cost, self.revenue_gen

	def get_holding_cost(self):
		return -self.CH * np.sum(self.observation)

	def get_order_cost(self, action):
		cost = 0
		total_per_product_amounts = action.sum(axis = 0)
		total_per_product_amounts = np.reshape(total_per_product_amounts, (total_per_product_amounts.shape[0], 1))
		cost = np.dot(self.CP_VECTOR.T, total_per_product_amounts)
		cost = np.squeeze(cost)
		# for whse_num in range(self.N_WHOUSES):
		# 	for prod_num in range(self.N_PRODUCTS):
		# 		mask = self.product_code_index_mapping['index'] == prod_num
		# 		CP = int(self.product_code_index_mapping.loc[mask].Cost)
		# 		cost += (CP * action[whse_num][prod_num])
		return -cost

	def render(self):
		t = PrettyTable(['W_House1', 'W_House2', 'W_House3', 'W_House4'])
		temp = self.observation.T
		
		for i in range(self.N_PRODUCTS):
			t.add_row(temp[i])

		print self.date, '*' * 25
		print 'Day Of The Year : ', self.day_of_year
		print t
		print 'Back-Order Costs : {} Holding Costs : {} Revenue : {} Order Costs : {}'.format(self.backorder_cost,self.holding_cost,self.revenue_gen, self.order_cost)


	def step(self, action):
		
		action = np.array(action)
		action[action < 0] = 0
		
		self.available, self.backorder_cost, self.revenue_gen = self.order_demand_deduction()
		self.action_q.append(action)
		self.observation = self.available
		self.holding_cost = self.get_holding_cost()
		self.order_cost = self.get_order_cost(action)
		self.reward = self.backorder_cost + self.holding_cost + self.revenue_gen + self.order_cost
		
		
		self.date += datetime.timedelta(days = 1) #Adding 1 day to date
		self.day_of_year = (self.day_of_year) % 365 + 1
		self.obs_flat = self.observation.flatten()
		self.obs_flat_date = np.hstack([self.obs_flat,np.array([self.day_of_year])])
		self.obs_flat_date = np.reshape(self.obs_flat_date, (self.obs_flat_date.shape[0],1))
		return self.obs_flat_date, np.squeeze(self.reward)