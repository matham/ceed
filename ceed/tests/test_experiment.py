import pytest
from .test_app.examples.experiment import set_serializer_even_count_bits
from ceed.storage.controller import DataSerializerBase


@pytest.mark.parametrize('n_sub_frames', [1, 4, 12])
def test_serializer_with_config(n_sub_frames):
    serializer = DataSerializerBase()
    config, num_handshake_ticks, counter, short_values, clock_values = \
        set_serializer_even_count_bits(serializer, n_sub_frames)
    serializer.projector_to_aquisition_map = {i: i for i in range(16)}

    bits = serializer.get_bits(config)
    next(bits)

    assert serializer.num_ticks_handshake(
        len(config), n_sub_frames) == num_handshake_ticks
    assert serializer.num_ticks_handshake(
        len(config), 1) * n_sub_frames == num_handshake_ticks

    i = 0
    for count, short, clock in zip(counter, short_values, clock_values):
        value = bits.send(i * n_sub_frames + 1)
        print(i, f'{value:010b}, {count:010b}, {short:010b}, {clock:08b}')
        assert value == count | short | clock

        i += 1

    assert i == len(counter)
