from htm_rl.mdp_agent.sar import Sar, SarSuperposition
from htm_rl.representations.int_sdr_encoder import IntSdrEncoder
from htm_rl.representations.sar_sdr_encoder import SarSdrEncoder as TSarSdrEncoder

SarSdrEncoder = TSarSdrEncoder[Sar, SarSuperposition, IntSdrEncoder]
