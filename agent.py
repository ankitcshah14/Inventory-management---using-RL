from environment import InventoryEnv

env = InventoryEnv()
env.reset()

for _ in range(365):
	env.render()
	action = env.action_sample()
	observation, reward = env.step(action)
	# print reward

