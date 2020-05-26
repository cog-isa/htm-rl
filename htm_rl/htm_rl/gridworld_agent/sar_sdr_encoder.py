from htm_rl.representations.sar_sdr_encoder import SarSdrEncoder
from htm_rl.gridworld_agent.list_sdr_encoder import ListSdrEncoder
from htm_rl.gridworld_agent.sar import Sar, SarSuperposition

SarSdrEncoder = SarSdrEncoder[Sar, SarSuperposition, ListSdrEncoder]
