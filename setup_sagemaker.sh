#shell script to setup the inventory environment on aws sagemaker
pip install gym
sudo yum update
sudo yum install openmpi-devel
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
cd
git clone https://github.com/chirag16/Inventory-management---using-RL
cd Inventory-management---using-RL/
cp inventory_env.py /home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.6/site-packages/gym/envs/toy_text/
cd /home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.6/site-packages/gym/envs/toy_text/
echo "from gym.envs.toy_text.inventory_env import InventoryEnv" >> __init__.py
cd ..
echo "register(id='InventoryEnv-v0',entry_point='gym.envs.toy_text:InventoryEnv',max_episode_steps=250,)" >> __init__.py
pip install prettytable
