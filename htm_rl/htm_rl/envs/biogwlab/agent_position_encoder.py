from htm_rl.common.sdr_encoders import IntBucketEncoder, SdrConcatenator


class AgentPositionEncoder:
    output_sdr_size: int

    include_view_direction: bool
    position_encoder: IntBucketEncoder
    view_direction_encoder: IntBucketEncoder
    sdr_concatenator: SdrConcatenator

    def __init__(self, n_positions, n_directions, include_view_direction, **state_encoder):
        self.position_encoder = IntBucketEncoder(n_values=n_positions, **state_encoder)
        self.output_sdr_size = self.position_encoder.output_sdr_size

        self.include_view_direction = include_view_direction
        if self.include_view_direction:
            self.view_direction_encoder = IntBucketEncoder(n_values=n_directions, **state_encoder)
            self.sdr_concatenator = SdrConcatenator([
                self.position_encoder, self.view_direction_encoder
            ])
            self.output_sdr_size = self.sdr_concatenator.output_sdr_size

    def encode(self, state):
        position, view_direction = state
        enc_position = self.position_encoder.encode(position)
        if not self.include_view_direction:
            return enc_position

        enc_view_direction = self.view_direction_encoder.encode(view_direction)
        return self.sdr_concatenator.concatenate(enc_position, enc_view_direction)