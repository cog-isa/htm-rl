#!/bin/zsh

# multi_way_v0
python ./run_mdp_tests.py -c ./experiments/paper/multi_way_v0.yml -g htm_0 dqn_greedy "htm_*_1g" -sp

python ./run_mdp_tests.py -c ./experiments/paper/multi_way_v0.yml -g htm_1_1g htm_1 -n 1 -sp

python ./run_mdp_tests.py -c ./experiments/paper/multi_way_v0.yml -g htm_2_1g htm_2 -n 2 -sp

python ./run_mdp_tests.py -c ./experiments/paper/multi_way_v0.yml -g htm_4_1g htm_4 -n 4 -sp


# multi_way_v2
python ./run_mdp_tests.py -c ./experiments/paper/multi_way_v2.yml -g htm_0 dqn_greedy "htm_*_1g" -sp

# python ./run_mdp_tests.py -c ./experiments/paper/multi_way_v2.yml -g htm_2_1g htm_2 -n 2 -sp

# python ./run_mdp_tests.py -c ./experiments/paper/multi_way_v2.yml -g htm_4_1g htm_4 -n 4 -sp

# python ./run_mdp_tests.py -c ./experiments/paper/multi_way_v2.yml -g htm_8_1g htm_8 -n 8 -sp
