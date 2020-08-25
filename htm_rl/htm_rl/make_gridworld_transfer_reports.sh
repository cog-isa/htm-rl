#!/bin/zsh

# multi_way_v0
python ./scratch.py -c ./experiments/multi_way_v0_transfer/config.yml -g htm_0 dqn_greedy "htm_*_1g" -s

python ./scratch.py -c ./experiments/multi_way_v0_transfer/config.yml -g htm_1_1g htm_1 -n 1 -s

python ./scratch.py -c ./experiments/multi_way_v0_transfer/config.yml -g htm_2_1g htm_2 -n 2 -s

python ./scratch.py -c ./experiments/multi_way_v0_transfer/config.yml -g htm_4_1g htm_4 -n 4 -s

python ./scratch.py -c ./experiments/multi_way_v0_transfer/config.yml -g dqn_greedy dqn_eps htm_4_1g -n dqn -s

rm ./experiments/multi_way_v0_transfer/*__*__rewards*
rm ./experiments/multi_way_v0_transfer/*__*__times*