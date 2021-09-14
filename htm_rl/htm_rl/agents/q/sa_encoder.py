from htm_rl.common.sdr import SparseSdr


class SaEncoder:
    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        raise NotImplementedError()

    def encode_action(self, action: int, learn: bool) -> SparseSdr:
        raise NotImplementedError()

    def concat_s_a(self, s: SparseSdr, a: SparseSdr, learn: bool) -> SparseSdr:
        raise NotImplementedError()

    def cut_s(self, s_a: SparseSdr) -> SparseSdr:
        raise NotImplementedError()

    def encode_s_a(self, s_a: SparseSdr, learn: bool) -> SparseSdr:
        raise NotImplementedError()

    # ----------- shortcuts ----------------
    def concat_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        raise NotImplementedError()

    def encode_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        raise NotImplementedError()

    @property
    def output_sdr_size(self):
        raise NotImplementedError()