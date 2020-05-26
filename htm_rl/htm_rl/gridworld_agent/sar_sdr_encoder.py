from htm_rl.representations.sar_sdr_encoder import SarSdrEncoder as TSarSdrEncoder
from htm_rl.gridworld_agent.list_sdr_encoder import ListSdrEncoder
from htm_rl.gridworld_agent.sar import Sar, SarSuperposition

SarSdrEncoder = TSarSdrEncoder[Sar, SarSuperposition, ListSdrEncoder]
