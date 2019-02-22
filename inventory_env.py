import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from prettytable import PrettyTable

class InventoryEnv(gym.Env):

    def __init__(self):
        self.N_WHOUSES, self.N_PRODUCTS = 4, 100
        self.action_space_low, self.action_space_high = 0, 10
        self.INV_L, self.INV_H = 0, 10

        self.low_state = np.zeros((self.N_WHOUSES * self.N_PRODUCTS,)) + self.INV_L
        self.high_state = np.zeros((self.N_WHOUSES * self.N_PRODUCTS,)) + self.INV_H

        self.action_space = spaces.Box(low=self.action_space_low, high=self.action_space_high,
                                       shape=(self.N_WHOUSES * self.N_PRODUCTS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)
        # Added by Chirag
        self.PROFIT_FACTOR = 5.     # Reduced the PROFIT_FACTOR from 5. to 1.
                                    # Reason: Ideally the agent must be able to 
                                    # perfectly predict the demand LEN_MONTH days 
                                    # from the current day. If this is true then
                                    # backorder_cost = holding_cost = 0, and 
                                    # order_cost = CP * action = 1 * action = action
                                    # revenue_gen = SP * demand = (1. + PROFIT_FACTOR) * CP * demand
                                    # but action = demand (sinc this is what the agent is supposed to do)
                                    # Thus, revenue_gen = (1. + PROFIT_FACTOR) * CP * action
                                    #                   = (1. + 1.) * 1. * action
                                    #                   = 2. * action
                                    # Thus, reward = revenue_gen - order_cost = 1 * action
                                    # This should be sufficient to overcome the initially incurred backorder_costs
                                    # Whereas if PROFIT_FACTOR was 5. then even if the agent successfully sells once,
                                    # the profits will overcome backorder_costs for many more steps.
                                    # Now the agent will not be as motivated to get rid of those backorder_costs.

                                    # Changed PROFIT_FACTOR back to 5. because the agent keeps taking action 1 when PROFIT_FACTOR is 1.
        self.CP = 1.                # For for loops
        self.CP_VECTOR = np.zeros((self.N_PRODUCTS, 1)) + self.CP # For vectorization
        self.SP_VECTOR = (1. + self.PROFIT_FACTOR) * self.CP_VECTOR

        self.CH = 1.
        self.LEN_MONTH = 10
        self.DEMAND_STD_DEV = 1
        self.DEMAND_MEAN = 5

        self.counter = 0
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def order_demand_deduction(self):
        if len(self.action_q) >= self.LEN_MONTH:
            self.available = self.observation + self.action_q.pop(0)
            for whse_num in range(self.N_WHOUSES):
                for prod_num in range(self.N_PRODUCTS):
                    if self.available[whse_num][prod_num] > self.INV_H:
                        self.available[whse_num][prod_num] = self.INV_H
        else:
            self.available = self.observation.copy()

        # Create Demand Matrix for Current Day
        self.backorder_cost = 0.
        self.revenue_gen = 0.


        demand_matrix = np.random.normal(self.DEMAND_MEAN, self.DEMAND_STD_DEV, size = (self.N_WHOUSES, self.N_PRODUCTS))
        demand_matrix = demand_matrix.astype(np.int32)
        self.dm = demand_matrix.copy()

        for whse_num in range(self.N_WHOUSES):
            for prod_num in range(self.N_PRODUCTS):
                SP = (1. + self.PROFIT_FACTOR) * self.CP
                if demand_matrix[whse_num][prod_num] > 0:
                    if self.available[whse_num][prod_num] >= demand_matrix[whse_num][prod_num]:
                        self.available[whse_num][prod_num] -= demand_matrix[whse_num][prod_num]
                        self.revenue_gen += SP * demand_matrix[whse_num][prod_num]
                    else:
                        self.backorder_cost += self.CP * (demand_matrix[whse_num][prod_num] - self.available[whse_num][prod_num])

        return self.available, -self.backorder_cost, self.revenue_gen

    def get_holding_cost(self):
        return -self.CH * np.sum(self.observation)

    def get_order_cost(self, action):
        cost = 0
        total_per_product_amounts = action.sum(axis = 0)
        total_per_product_amounts = np.reshape(total_per_product_amounts, (total_per_product_amounts.shape[0], 1))
        cost = np.dot(self.CP_VECTOR.T, total_per_product_amounts)
        cost = np.squeeze(cost)

        return -cost

    def step(self, action):
        action = np.array(action).astype(np.int32)
        action = action.reshape(self.N_WHOUSES, self.N_PRODUCTS)
        action[action < 0] = 0
        
        self.available, self.backorder_cost, self.revenue_gen = self.order_demand_deduction()
        self.action_q.append(action)
        self.observation = self.available.copy()
        self.holding_cost = self.get_holding_cost()
        self.order_cost = self.get_order_cost(action)
        self.reward = self.backorder_cost + self.holding_cost + self.revenue_gen + self.order_cost 
        self.reward /= 100.
        self.reward /= self.N_WHOUSES * self.N_PRODUCTS
        
        # print (self.reward)
        # if self.counter == 1: 
        #     print('Action: {}\tDemand: {}'.format(action[0][0], self.dm[0][0]))
        # if self.revenue_gen > 0:
        # print (self.backorder_cost, self.revenue_gen, self.holding_cost, self.order_cost, self.reward)

        self.counter += 1
        return np.array(self.observation).flatten(), np.squeeze(self.reward), False, None

    def reset(self):
        self.action_q = []
        self.order_cost = 0
        self.holding_cost = 0
        self.backorder_cost = 0
        self.revenue_gen = 0
        self.available = np.zeros((self.N_WHOUSES, self.N_PRODUCTS))
        self.observation = np.zeros((self.N_WHOUSES, self.N_PRODUCTS))
        self.counter = 0
        return np.array(self.observation).flatten()

    def render(self, mode='human'):
        display_table = PrettyTable(['W_House{}'.format(i) for i in range(self.N_WHOUSES)])
        temp = self.observation.T

        for i in range(self.N_PRODUCTS):
            display_table.add_row(temp[i])

        print('*' * 50)
        print(display_table)
        print('Back-Order Costs : {} Holding Costs : {} Revenue : {} Order Costs : {}'.format(self.backorder_cost,self.holding_cost,self.revenue_gen, self.order_cost))

    def close(self):
        pass
