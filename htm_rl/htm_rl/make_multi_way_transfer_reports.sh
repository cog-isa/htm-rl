#!/bin/zsh

# multi_way_v0
python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v0.yml -g htm_0 dqn_greedy "htm_*_1g" -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v0.yml -g htm_1_1g htm_1 -n 1 -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v0.yml -g htm_2_1g htm_2 -n 2 -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v0.yml -g htm_4_1g htm_4 -n 4 -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v0.yml -g dqn_greedy dqn_eps htm_4_1g -n dqn -s

rm ./experiments/multi_way_transfer/*__*__rewards*
rm ./experiments/multi_way_transfer/*__*__times*


# multi_way_v1
python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v1.yml -g htm_0 dqn_greedy "htm_*_1g" -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v1.yml -g htm_1_1g htm_1 -n 1 -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v1.yml -g htm_2_1g htm_2 -n 2 -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v1.yml -g htm_4_1g htm_4 -n 4 -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v1.yml -g dqn_greedy dqn_eps htm_4_1g -n dqn -s

rm ./experiments/multi_way_transfer/*__*__rewards*
rm ./experiments/multi_way_transfer/*__*__times*


# multi_way_v2
python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v2.yml -g htm_0 dqn_greedy "htm_*_1g" -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v2.yml -g htm_2_1g htm_2 -n 2 -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v2.yml -g htm_4_1g htm_4 -n 4 -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v2.yml -g htm_8_1g htm_8 "htm_12*" -n "8-12" -s

python ./run_mdp_tests.py -c ./experiments/multi_way_transfer/multi_way_v2.yml -g dqn_greedy dqn_eps htm_4_1g -n dqn -s

rm ./experiments/multi_way_transfer/*__*__rewards*
rm ./experiments/multi_way_transfer/*__*__times*
