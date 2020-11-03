#!/bin/zsh

# # env examples
# python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_1_1_200_1337.yml -d 3
# rm ./experiments/paper/gridworld_5x5_1_1_200_1337_map_1_*.svg
# python ./run_mdp_tests.py -c ./experiments/paper/gridworld_8x8_10_1_10_1337.yml -d 8
# rm ./experiments/paper/gridworld_8x8_10_1_10_1337_map_1_*.svg
# rm ./experiments/paper/gridworld_8x8_10_1_10_1337_map_2_*.svg
# rm ./experiments/paper/gridworld_8x8_10_1_10_1337_map_3_*.svg
# rm ./experiments/paper/gridworld_8x8_10_1_10_1337_map_4_*.svg
# rm ./experiments/paper/gridworld_8x8_10_1_10_1337_map_5_*.svg
# rm ./experiments/paper/gridworld_8x8_10_1_10_1337_map_6_*.svg


# gridworld_5x5_100_2_8_1337
python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_100_2_8_1337.yml -g htm_0 dqn_greedy "htm_*_1g" -sp

python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_100_2_8_1337.yml -g dqn_greedy dqn_eps htm_4_1g -n dqn -sp

python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_100_2_8_1337.yml -g "htm_2_*g" -n 2 -sp

python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_100_2_8_1337.yml -g "htm_4_*g" htm_8_1g -n "4-8" -sp


# gridworld_5x5_20_1_20_1337
python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_20_1_20_1337.yml -g htm_0 dqn_greedy "htm_*_1g" -sp

python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_20_1_20_1337.yml -g dqn_greedy dqn_eps htm_4_1g -n dqn -sp

python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_20_1_20_1337.yml -g htm_2_1g htm_2_4g htm_2_16g -n 2 -sp

python ./run_mdp_tests.py -c ./experiments/paper/gridworld_5x5_20_1_20_1337.yml -g htm_4_1g htm_4_4g htm_4_16g htm_8_1g -n "4-8" -sp
