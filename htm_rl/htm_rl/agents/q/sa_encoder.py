from htm_rl.common.sdr import SparseSdr


class SaEncoder:
    """
    State, action encoder.
    """

    def encode_state(self, state: SparseSdr, learn: bool) -> SparseSdr:
        """
        Encodes state sparse sdr to another state sparse sdr (aka s):
            state --> s
        """
        raise NotImplementedError()

    def encode_action(self, action: int, learn: bool) -> SparseSdr:
        """
        Encodes action int to action sparse sdr (aka a):
            action --> a
        """
        raise NotImplementedError()

    def concat_s_a(self, s: SparseSdr, a: SparseSdr, learn: bool) -> SparseSdr:
        """
        Encodes state (s) and action (a) sparse sdrs to joined representation,
        such that it can be decoded back (hence the name "concat" as it's
        the most naive implementation). The naming for such representation is s_a:
            s, a --> s_a
        """
        raise NotImplementedError()

    def cut_s(self, s_a: SparseSdr) -> SparseSdr:
        """
        Decodes joined decodable state-action sparse sdr representation s_a,
        returning only state sparse sdr s:
            s_a --> s
        """
        raise NotImplementedError()

    def encode_s_a(self, s_a: SparseSdr, learn: bool) -> SparseSdr:
        """
        Encodes joined decodable state-action sparse sdr representation s_a
        to another [possibly not decodable] joined sparse sdr representation sa:
            s_a --> sa
        """
        raise NotImplementedError()

    # ----------- shortcuts ----------------
    def concat_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        """
        Encodes state (s) sparse sdrs and action int to joined decodable
        state-action sparse sdr representation s_a:
            s, action --> s_a
        It's a shortcut method, which combines following (but could be implemented
        faster):
            action --> a
            s, a --> s_a
        """
        raise NotImplementedError()

    def encode_s_action(self, s: SparseSdr, action: int, learn: bool) -> SparseSdr:
        """
        Encodes state (s) sparse sdrs and action int to [possibly not decodable]
        joined sparse sdr representation sa:
            s, action --> sa
        It's a shortcut method, which combines following (but could be implemented
        faster):
            action --> a
            s, a --> s_a
            s_a --> sa
        """
        raise NotImplementedError()

    @property
    def output_sdr_size(self):
        """
        Represent the size of the joined state-action sparse sdr
        representation (sa).
        """
        raise NotImplementedError()