"""Ceed-MCS data merging
=========================

The classes to merge and align the Ceed and MCS recording data into a single
file.
"""

import sys
import datetime
from tqdm import tqdm
from collections import defaultdict
import os.path
from math import ceil
from typing import List, Dict, Optional
from shutil import copy2
from McsPy import ureg
import McsPy.McsData
import numpy as np
import nixio as nix
from more_kivy_app.utils import yaml_dumps, yaml_loads

__all__ = ('CeedMCSDataMerger', 'DigitalDataStore', 'MCSDigitalData',
           'CeedDigitalData', 'AlignmentException')

McsPy.McsData.VERBOSE = False


class AlignmentException(Exception):
    pass


class BitMapping32:

    l_to_l: np.ndarray = None

    h_to_l: np.ndarray = None

    l_to_h: np.ndarray = None

    h_to_h: np.ndarray = None

    def __init__(self, bits: List[int]):
        self._compute_maps(bits)

    def _compute_maps(self, bits: List[int]):
        indices = np.arange(2 ** 16, dtype=np.uint16)

        l_to_l = np.zeros((2 ** 16, ), dtype=np.uint16)
        h_to_l = np.zeros((2 ** 16, ), dtype=np.uint16)
        l_to_h = np.zeros((2 ** 16, ), dtype=np.uint16)
        h_to_h = np.zeros((2 ** 16, ), dtype=np.uint16)

        for i, k in enumerate(bits):
            if i <= 15:
                if k <= 15:
                    l_to_l |= ((indices & (1 << k)) >> k) << i
                else:
                    h_to_l |= \
                        (indices & (1 << (k % 16))) >> (k % 16) << i
            else:
                if k <= 15:
                    l_to_h |= ((indices & (1 << k)) >> k) << i
                else:
                    h_to_h |= \
                        (indices & (1 << (k % 16))) >> (k % 16) << i

        self.l_to_l = l_to_l
        self.h_to_l = h_to_l
        self.l_to_h = l_to_h
        self.h_to_h = h_to_h

    def map(self, data: np.ndarray) -> np.ndarray:
        mapped = np.zeros(len(data), dtype=np.uint32)
        # get upper bits and shift up
        mapped |= self.l_to_h[data & 0xFFFF] | \
            self.h_to_h[(data >> 16) & 0xFFFF]
        mapped <<= 16
        # get lower bits
        mapped |= self.l_to_l[data & 0xFFFF] | \
            self.h_to_l[(data >> 16) & 0xFFFF]

        return mapped


class DigitalDataStore:

    counter_bit_width: int = 32

    short_map: BitMapping32 = None

    short_count_indices: List[int] = None

    short_count_data: np.ndarray = None

    count_map: BitMapping32 = None

    count_indices: List[int] = None

    count_data: np.ndarray = None

    clock_index: int = None

    clock_data: np.ndarray = None

    data: np.ndarray = None
    '''raw data.

    Only includes rendered frame data. Skipped frames are not included.

    maps the values from data at indices_center using the maps at xxx_map
    for each array item and saves it into xxx_data. xxx_map is formed from
    xxx_indices.

    data_indices_xxx are the indices in data where each xxx_data array item
    came from.
    '''

    expected_handshake_len: int = 0

    handshake_data: bytes = b''

    counter: np.ndarray = None

    def __init__(self, short_count_indices, count_indices, clock_index,
                 counter_bit_width):
        '''indices are first mapped with projector_to_aquisition_map. '''
        super(DigitalDataStore, self).__init__()
        self.short_count_indices = short_count_indices
        self.count_indices = count_indices
        self.clock_index = clock_index
        self.counter_bit_width = counter_bit_width

        self.short_map = BitMapping32(short_count_indices)
        self.count_map = BitMapping32(count_indices)

    def _parse_components(self, data: np.ndarray) -> None:
        self.data = data
        clock_index = self.clock_index
        clock_bit = 1 << clock_index

        self.clock_data = (data & clock_bit) >> clock_index
        self.short_count_data = self.short_map.map(data)
        self.count_data = self.count_map.map(data)

    def compare(
            self, short_count_indices, count_indices, clock_index):
        if self.short_count_indices is None and \
                short_count_indices is not None:
            return False
        if self.short_count_indices is not None \
                and short_count_indices is None:
            return False
        if self.short_count_indices is not None:
            if not np.array_equal(
                    self.short_count_indices, short_count_indices):
                return False

        if self.count_indices is None and count_indices is not None:
            return False
        if self.count_indices is not None and count_indices is None:
            return False
        if self.count_indices is not None:
            if not np.array_equal(self.count_indices, count_indices):
                return False

        if self.clock_index != clock_index:
            return False
        return True

    def __eq__(self, other):
        if not isinstance(other, DigitalDataStore):
            return False

        return self.compare(
            other.short_count_indices, other.count_indices, other.clock_index)

    @property
    def n_parts_per_int(self):
        return int(
            ceil(self.counter_bit_width / len(self.count_indices)))

    @property
    def n_bytes_per_int(self):
        return self.counter_bit_width // 8

    def get_count_ints(self, count_data, contains_sub_frames, n_sub_frames):
        """Counter with sub-frames removed.
        """
        # last value in count_data could be spurious (from mcs)
        n_counter_bits = len(self.count_indices)
        n_parts_per_int = self.n_parts_per_int
        # all sub frames are the same (if we have them)
        # We don't need inverted frames
        count = count_data
        if contains_sub_frames:
            count = count_data[::n_sub_frames]

        count_inverted_2d = count[1::2]
        count = count[::2]

        # only keep full ints
        remainder = len(count) % n_parts_per_int
        if remainder:
            count = count[:-remainder]
        remainder = len(count_inverted_2d) % n_parts_per_int
        if remainder:
            count_inverted_2d = count_inverted_2d[:-remainder]

        # it comes in little endian as ints of counter_bit_width size
        count = np.reshape(count, (-1, n_parts_per_int))
        count_inverted_2d = np.reshape(count_inverted_2d, (-1, n_parts_per_int))
        # should have same height
        n = min(count.shape[0], count_inverted_2d.shape[0])
        count = count[:n, :]
        count_inverted_2d = count_inverted_2d[:n, :]
        assert count.shape == count_inverted_2d.shape

        # if inverted has one row less than count, chop it off
        count_2d = count.copy()
        if count_2d.shape[0] != count_inverted_2d.shape[0]:
            count_2d = count_2d[:-1, :]

        # we use 64 bit as max value for counter size
        count = count.astype(np.uint64)
        for i in range(n_parts_per_int):
            count[:, i] <<= i * n_counter_bits
        count = np.bitwise_or.reduce(count, axis=1)

        return count, count_2d, count_inverted_2d

    def check_counter_consistency(
            self, count_data, count_2d, count_inverted_2d, n_handshake_ints,
            contains_sub_frames, n_sub_frames, exclude_last_value):
        if count_2d.shape[0]:
            # even if no handshake sent, the first value will be size (zero)
            # so we always have at least one item
            assert n_handshake_ints
        count_shape = count_2d.shape

        if contains_sub_frames:
            count_data = count_data[::n_sub_frames]
        # remove data already in count_2d
        count_data = count_data[count_shape[0] * count_shape[1] * 2:]
        count_end = has_trailing = len(count_data)
        # we can't check the last item if it doesn't have inverted value
        if count_end % 2:
            count_end -= 1
        elif exclude_last_value:
            count_end -= 2
            count_end = max(count_end, 0)

        # set count bits
        count_bits = 0
        for i in range(len(self.count_indices)):
            count_bits |= 1 << i

        # do we have data beyond the handshake
        has_counter = count_shape[0] > n_handshake_ints
        # if exclude final, see if data extends past handshake
        handshake_end = n_handshake_ints
        if not has_trailing and not has_counter and exclude_last_value:
            handshake_end -= 1

        # if exclude final, see if data extends past counter
        count_2d_end = count_shape[0]
        if not has_trailing and exclude_last_value:
            count_2d_end -= 1

        # config is not inverted
        if not np.all(count_2d[:handshake_end, :]
                      == count_inverted_2d[:handshake_end, :]):
            raise AlignmentException('Handshake data corrupted')

        # first item is not inverted
        if not np.all(count_2d[:count_2d_end, 0]
                      == count_inverted_2d[:count_2d_end, 0]):
            raise AlignmentException('Non-inverted data group corrupted')

        # remaining counter is inverted
        if not np.all(
                (count_2d[n_handshake_ints:count_2d_end, 1:] ^ count_bits)
                & count_bits
                == count_inverted_2d[n_handshake_ints:count_2d_end, 1:]):
            raise AlignmentException('Inverted data group corrupted')

        # n_handshake_ints >= 1, if we have at least one full int
        if not n_handshake_ints:
            # only have partial int, handshake is not inverted
            assert not count_shape[0]
            if not np.all(
                    count_data[:count_end:2] == count_data[1:count_end:2]):
                raise AlignmentException('Handshake length data corrupted')
        elif count_shape[0] < n_handshake_ints:
            # have handshake size and remaining is handshake item, not inverted
            if not np.all(
                    count_data[:count_end:2] == count_data[1:count_end:2]):
                raise AlignmentException('Remaining handshake data corrupted')
        else:
            # remaining is regular data, first val is not inverted, the rest is
            if count_end:
                # have at least 2 values (count_end is multiple of 2)
                if count_data[0] != count_data[1]:
                    raise AlignmentException(
                        'Remaining handshake data corrupted')
            if not np.all((count_data[2:count_end:2] ^ count_bits) & count_bits
                          == count_data[3:count_end:2]):
                raise AlignmentException(
                    'Remaining inverted handshake data corrupted')

    def get_handshake(
            self, count: np.ndarray, contains_sub_frames, n_sub_frames):
        """If handshake is incomplete returns empty.
        """
        if len(count):
            n_config = int(count[0])
            # sanity check
            if n_config > 50:
                raise AlignmentException(
                    f'Got too many ({n_config}) handshake numbers')

            # it's in little endian so it doesn't need anything special
            # but we need to remove extra bytes given we use 64 bit (the max)
            config = b''.join(
                [b.tobytes()[:self.n_bytes_per_int]
                 for b in count[1:1 + n_config]])
            n_config_frames = (n_config + 1) * 2 * self.n_parts_per_int
            if contains_sub_frames:
                n_config_frames *= n_sub_frames
            return config, n_config * self.n_bytes_per_int, n_config + 1, \
                n_config_frames

        return b'', 0, 0, 0

    def check_missing_frames(
            self, short_count_data, contains_sub_frames, n_sub_frames):
        short = short_count_data
        if contains_sub_frames:
            short = short[::n_sub_frames]
        max_shot_val = 2 ** len(self.short_count_indices)

        i = short[0]
        for k, val in enumerate(short):
            if i != val:
                raise AlignmentException(f'Skipped a frame at frame {k}')
            i += 1
            if i == max_shot_val:
                i = 0


class MCSDigitalData(DigitalDataStore):

    data_indices_start: np.ndarray = None

    data_indices_end: np.ndarray = None

    experiments: Dict[bytes, List[tuple]]

    def parse(
            self, ceed_version, data, t_start, f,
            find_start_from_ceed_time=False, estimated_start: float = 0,
            pre_estimated_start: float = 0):
        self._parse_components(data)
        self.reduce_samples(
            t_start, f, find_start_from_ceed_time,
            estimated_start, pre_estimated_start)

        if ceed_version == '1.0.0.dev0':
            return
        self.parse_experiments()

    def reduce_samples(
            self, t_start, f, find_start_from_ceed_time=False,
            estimated_start: float = 0, pre_estimated_start: float = 0):
        """Reduces the data from multiple samples per-frame, to one sample per
        frame, according to the clock.
        """
        # data is already converted to normal lower bits
        clock_data = self.clock_data
        short_count_data = self.short_count_data
        count_data = self.count_data

        offset = 0
        if find_start_from_ceed_time:
            offset = (estimated_start - t_start).total_seconds() - \
                float(pre_estimated_start)
            if offset < 0:
                raise ValueError(
                    'Ceed data is not in the mcs data, given the offset')

            offset = int(offset * f)
            clock_data = clock_data[offset:]
            short_count_data = short_count_data[offset:]
            count_data = count_data[offset:]

        # should have at least 10 samples. At 5k sampling rate it's reasonable
        if len(clock_data) < 10:
            raise TypeError(
                'There is not enough data in the mcs file to be able to align '
                'with Ceed')

        clock_change = np.argwhere(clock_data[1:] - clock_data[:-1]).squeeze()
        # indices in data where value is different from last (including 0)
        idx_start = np.array([0], dtype=clock_change.dtype)
        # indices in data where next value is different (including last value)
        idx_end = np.array([len(clock_data) - 1], dtype=clock_change.dtype)
        if len(clock_change):
            idx_start = np.concatenate((idx_start, clock_change + 1))
            idx_end = np.concatenate((clock_change, idx_end))

        # take value after clock changes
        indices = np.minimum(idx_end - idx_start, 1) + idx_start

        # start at the
        s = 0 if clock_data[0] else 1
        indices = indices[s:]
        idx_start = idx_start[s:]
        idx_end = idx_end[s:]

        # indices in the original data
        self.data_indices_start = idx_start + offset
        self.data_indices_end = idx_end + offset
        # condensed data
        self.clock_data = clock_data[indices]
        self.short_count_data = short_count_data[indices]
        self.count_data = count_data[indices]

    def parse_experiments(self):
        # assuming the experiments recorded have at least two good frames,
        # otherwise we can't estimate expected frame size
        if self.n_parts_per_int <= 1:
            raise NotImplementedError(
                'Must break counter int into at least two parts so we can '
                'locate clock inverted values')

        max_shot_val = 2 ** len(self.short_count_indices)
        count_data_full = self.count_data
        short_count_data_full = self.short_count_data
        clock_data_full = self.clock_data
        start = self.data_indices_start
        end = self.data_indices_end
        diff = end - start
        med = np.median(diff)
        # each experiment is proceeded by 30-50 blank frames, so 10 is safe.
        # And we should never skip 10+ frames sequentially in a stable system
        breaks = np.nonzero(diff >= (10 * med))[0]

        experiments = self.experiments = defaultdict(list)
        start_i = 0
        for break_i in breaks:
            s = start_i
            # the long frame is included in last experiment. If long frame is
            # clock low, first frame of next exp is high. If it's high, clock
            # goes low for some frames and then high, so high frame will be
            # first short frame
            e = break_i + 1
            # get section of this possible experiment
            count_data = count_data_full[s:e]
            short_count_data = short_count_data_full[s:e]
            clock_data = clock_data_full[s:e]
            start_i = e

            # need some data to work with
            if len(count_data) < 4:
                continue
            # need to start high
            if not clock_data[0]:
                count_data = count_data[1:]
                short_count_data = short_count_data[1:]
                s += 1

            try:
                # use short counter to see if missing frames, exclude final
                self.check_missing_frames(short_count_data[:-1], False, 1)

                # we don't drop last frame, but the frame extends too long post
                # experiment (i.e. last was clock low and it stayed clock low
                # until next experiment). And last item may be spurious
                end[e - 1] = start[e - 1] + med

                # chop off partial ints and get full ints, it's ok if last value
                # is spurious
                count, count_2d, count_inverted_2d = self.get_count_ints(
                    count_data, False, 1)
                # get handshake from full ints. Last val could be spurious, so
                # if it's part of the handshake, handshake is not complete
                handshake_data, handshake_len, n_handshake_ints, \
                    n_config_frames = self.get_handshake(count, False, 1)
                # check that full and partial ints match
                self.check_counter_consistency(
                    count_data, count_2d, count_inverted_2d, n_handshake_ints,
                    False, 1, True)
            except AlignmentException:
                continue

            if not handshake_len:
                continue

            # the last count or handshake value could be spurious, but then it
            # won't match, which is ok because we need the full handshake and
            # when searching for handshake we anyway chop of end until empty
            # or found
            experiments[handshake_data].append((
                start[s:e], end[s:e], handshake_len, count_data, count))


class CeedDigitalData(DigitalDataStore):

    def parse(
            self, ceed_version, frame_bits: np.ndarray,
            frame_counter: np.ndarray, start_t: np.ndarray, n_sub_frames: int,
            rendered_frames: np.ndarray
    ) -> None:
        if ceed_version == '1.0.0.dev0':
            self.parse_data_v1_0_0_dev0(frame_bits)
        else:
            self.parse_data(
                frame_bits, frame_counter, start_t, n_sub_frames,
                rendered_frames)

    def parse_data_v1_0_0_dev0(self, data: np.ndarray) -> None:
        self._parse_components(data)

    def parse_data(
            self, frame_bits: np.ndarray, frame_counter: np.ndarray,
            start_t: np.ndarray, n_sub_frames: int, rendered_frames: np.ndarray
    ) -> None:
        # the last frame(s) may or may not have been rendered (e.g. with
        # sub-frames, all the sub-frames may not have been rendered)
        contains_sub_frames = True
        self._parse_components(frame_bits[rendered_frames])

        # use short counter to see if missing frames
        self.check_missing_frames(
            self.short_count_data, contains_sub_frames, n_sub_frames)
        # chop off partial ints and get full ints
        count, count_2d, count_inverted_2d = self.get_count_ints(
            self.count_data, contains_sub_frames, n_sub_frames)
        # get handshake from full ints
        handshake_data, handshake_len, n_handshake_ints, n_config_frames = \
            self.get_handshake(count, contains_sub_frames, n_sub_frames)
        # check that full and partial ints match
        self.check_counter_consistency(
            self.count_data, count_2d, count_inverted_2d, n_handshake_ints,
            contains_sub_frames, n_sub_frames, False)

        self.handshake_data = handshake_data
        self.expected_handshake_len = handshake_len
        self.counter = frame_counter[rendered_frames].copy()


class CeedMCSDataMerger:

    ceed_filename: str = ''

    ceed_global_config = {}

    ceed_version: str = ''

    ceed_config_orig = {}

    ceed_data = {}

    ceed_data_container = None

    mcs_filename: str = ''

    mcs_dig_data = []

    mcs_dig_config = {}

    mcs_data_container = None

    def __init__(self, ceed_filename, mcs_filename):
        self.ceed_filename = ceed_filename
        self.mcs_filename = mcs_filename

    @property
    def n_sub_frames(self):
        """Specific to each experiment and can change between them.
        """
        video_mode = self.ceed_config_orig['view']['video_mode']
        n_sub_frames = 1
        if video_mode == 'QUAD4X':
            n_sub_frames = 4
        elif video_mode == 'QUAD12X':
            n_sub_frames = 12
        return n_sub_frames

    def get_experiment_numbers(self, ignore_list=None):
        from ceed.storage.controller import CeedDataWriterBase
        nix_file = nix.File.open(self.ceed_filename, nix.FileMode.ReadOnly)
        try:
            names = CeedDataWriterBase.get_blocks_experiment_numbers(
                nix_file.blocks, ignore_list)
        finally:
            nix_file.close()
        return names

    def read_mcs_data(self):
        filename = self.mcs_filename
        self.mcs_data_container = None

        data = McsPy.McsData.RawData(filename)
        if not len(data.recordings):
            raise Exception('There is no data in {}'.format(filename))
        if len(data.recordings) > 1:
            raise Exception('There is more than one recording in {}'.
                            format(filename))
        t_start = data.date

        chan = f = None
        for stream in data.recordings[0].analog_streams:
            if 128 in data.recordings[0].analog_streams[stream].channel_infos:
                chan = stream
                f = data.recordings[0].analog_streams[stream].\
                    channel_infos[128].sampling_frequency
                f = f.m_as(ureg.hertz)
                break
        if chan is None:
            raise Exception('Did not find digital data channel')

        self.mcs_dig_data = np.array(
            data.recordings[0].analog_streams[chan].channel_data).squeeze()
        self.mcs_dig_config = {'t_start': t_start, 'f': f}

    def read_ceed_data(self):
        from ceed.analysis import read_nix_prop
        filename = self.ceed_filename
        self.ceed_data_container = None

        f = nix.File.open(filename, nix.FileMode.ReadOnly)
        try:
            config_data = {}
            for prop in f.sections['app_config'].props:
                config_data[prop.name] = yaml_loads(read_nix_prop(prop))
        finally:
            f.close()

        self.ceed_global_config = config_data['app_settings']
        self.ceed_version = config_data['ceed_version']

    def read_ceed_experiment_data(self, experiment):
        # If there's unrendered frames (except last), it raises error. Removes
        # last frame if it's not rendered
        from ceed.storage.controller import CeedDataWriterBase
        from ceed.analysis import read_nix_prop
        if not self.ceed_global_config:
            raise TypeError(
                'Global ceed data not read. Please first call read_ceed_data')

        filename = self.ceed_filename
        f = nix.File.open(filename, nix.FileMode.ReadOnly)

        block = f.blocks[
            CeedDataWriterBase.get_experiment_block_name(experiment)]
        section = block.metadata
        start_t = datetime.datetime(1970, 1, 1) + \
            datetime.timedelta(seconds=block.created_at)

        metadata = {}
        try:
            try:
                config = section.sections['app_config']
            except KeyError as exc:
                raise KeyError(
                    'Did not find config in experiment info for experiment {}'.
                    format(experiment)) from exc

            for prop in config:
                metadata[prop.name] = yaml_loads(read_nix_prop(prop))
            self.ceed_config_orig = metadata['app_settings']
            # ceed_config_orig must be set to read n_sub_frames
            n_sub_frames = self.n_sub_frames
            skip = self.ceed_config_orig['view'].get(
                'skip_estimated_missed_frames', False)

            if not block.data_arrays['frame_bits'].shape or \
                    not block.data_arrays['frame_bits'].shape[0]:
                raise Exception('Experiment {} has no data'.format(experiment))

            frame_bits = np.asarray(block.data_arrays['frame_bits']).squeeze()
            frame_counter = np.asarray(
                block.data_arrays['frame_counter']).squeeze()

            # rendered_counter is multiples of n_sub_frames, starting from
            # n_sub_frames. Missed frames don't have number in rendered_counter
            rendered_counter = np.asarray(
                block.data_arrays['frame_time_counter']).squeeze()
            if skip:
                count_indices = np.arange(1, 1 + len(frame_counter))
                found = rendered_counter[:, np.newaxis] - \
                    np.arange(n_sub_frames)[np.newaxis, :]
                found = found.reshape(-1)
                rendered_frames = np.isin(count_indices, found)
            else:
                if not np.all(
                        rendered_counter == np.arange(
                            n_sub_frames, len(frame_counter) + 1, n_sub_frames)
                ):
                    raise ValueError(
                        'Some frames were not rendered and skipped')

                rendered_frames = np.ones(len(frame_counter), dtype=np.bool)

        except Exception:
            f.close()
            raise
        else:
            f.close()

        self.ceed_data = {
            'frame_bits': frame_bits, 'frame_counter': frame_counter,
            'start_t': start_t, 'rendered_frames': rendered_frames}

    def create_or_reuse_ceed_data_container(self):
        config = self.ceed_config_orig['serializer']
        short_count_indices = config['short_count_indices']
        count_indices = config['count_indices']
        clock_index = config['clock_idx']
        counter_bit_width = config['counter_bit_width']

        if self.ceed_data_container is None:
            self.ceed_data_container = CeedDigitalData(
                short_count_indices, count_indices, clock_index,
                counter_bit_width)
        elif not self.ceed_data_container.compare(
                short_count_indices, count_indices, clock_index):
            raise ValueError('Ceed-MCS hardware mapping has changed in file')
        return self.ceed_data_container

    def create_or_reuse_mcs_data_container(self):
        config = self.ceed_global_config['serializer']
        ceed_mcs_map = config['projector_to_aquisition_map']

        short_count_indices = [
            ceed_mcs_map[i] for i in config['short_count_indices']]
        count_indices = [ceed_mcs_map[i] for i in config['count_indices']]
        clock_index = ceed_mcs_map[config['clock_idx']]
        counter_bit_width = config['counter_bit_width']

        if self.mcs_data_container is None:
            self.mcs_data_container = MCSDigitalData(
                short_count_indices, count_indices, clock_index,
                counter_bit_width)
        elif not self.mcs_data_container.compare(
                short_count_indices, count_indices, clock_index):
            raise ValueError('Ceed-MCS hardware mapping has changed in file')
        return self.mcs_data_container

    def parse_ceed_experiment_data(self):
        if not self.ceed_data:
            raise TypeError(
                'Ceed experiment data not read. Please first call '
                'read_ceed_experiment_data')

        n_sub_frames = self.n_sub_frames
        self.create_or_reuse_ceed_data_container()
        self.ceed_data_container.parse(
            self.ceed_version, **self.ceed_data, n_sub_frames=n_sub_frames)

    def parse_mcs_data(
            self, find_start_from_ceed_time: bool = False,
            pre_estimated_start: float = 0, estimated_start: float = 0):
        """estimated_start should be ``ceed_data['start_t']`` if used.
        """
        if not self.mcs_dig_config:
            raise TypeError(
                'MCS data not read. Please first call read_mcs_data')
        if not self.ceed_global_config:
            raise TypeError(
                'Global ceed data not read. Please first call read_ceed_data')

        self.create_or_reuse_mcs_data_container()
        self.mcs_data_container.parse(
            self.ceed_version, self.mcs_dig_data,
            self.mcs_dig_config['t_start'], self.mcs_dig_config['f'],
            find_start_from_ceed_time=find_start_from_ceed_time,
            pre_estimated_start=pre_estimated_start,
            estimated_start=estimated_start)

    def get_alignment(self):
        if self.ceed_version == '1.0.0.dev0':
            return self._get_alignment_v1_0_0_dev0(True)
        return self._get_alignment(True)

    def _get_alignment(self, search_uuid=True):
        if not search_uuid:
            raise NotImplementedError

        ceed_ = self.ceed_data_container
        mcs = self.mcs_data_container
        handshake = ceed_.handshake_data
        if not handshake:
            raise AlignmentException(
                'Cannot find experiment - no Ceed experiment ID parsed')

        if handshake not in mcs.experiments:
            if not mcs.experiments:
                raise AlignmentException(
                    'Cannot find any experiment in the MCS parsed data')

            while handshake and handshake not in mcs.experiments:
                handshake = handshake[:-1]

            if not handshake:
                raise AlignmentException(
                    'Cannot find experiment in the MCS parsed data')

        experiments = mcs.experiments[handshake]
        if len(experiments) != 1:
            raise AlignmentException(
                'Found more than one matching experiment in MCS data, '
                'experiment was likely stopped before the full Ceed-MCS '
                'handshake completed')

        # the last count or handshake value could be spurious, but then it
        # won't match, which is ok because we need the full handshake
        start, end, handshake_len, count_data, count = experiments[0]
        # n_sub_frames can change between experiments
        n_sub_frames = self.n_sub_frames
        # count_data is same for all sub-frames, but counter increments
        ceed_count_data = ceed_.count_data

        # ceed counter contains an item for each frame and sub-frame
        assert not len(ceed_count_data) % n_sub_frames
        # mcs only sees frames, because sub-frames are all the same
        ceed_count_data_main_frames = ceed_count_data[::n_sub_frames]
        n_ceed = len(ceed_count_data_main_frames)
        n_mcs = len(count_data)
        assert n_mcs

        if n_mcs < n_ceed:
            raise AlignmentException(
                'MCS missed some ceed frames, cannot align')
        if n_mcs > n_ceed + 1:
            raise AlignmentException(
                'MCS read frames that ceed did not send, cannot align')

        if n_mcs != n_ceed:
            # last frame could be spurious on the mcs side
            n_mcs -= 1

        count_data = count_data[:n_mcs]
        if not np.all(ceed_count_data_main_frames == count_data):
            raise AlignmentException(
                'Counter data itemds does not match between Ceed and MCS')

        start = start[:n_mcs]
        end = end[:n_mcs]
        if n_sub_frames == 1:
            return start

        n_frames = end - start + 1
        n_frames = n_frames[:, np.newaxis] / n_sub_frames
        split_frames = np.arange(n_sub_frames)[np.newaxis, :]
        start = np.round(split_frames * n_frames + start[:, np.newaxis])
        start = np.asarray(start, dtype=np.int64).reshape(-1)

        return start

    def _get_alignment_v1_0_0_dev0(self, search_uuid=True):
        ceed_ = self.ceed_data_container
        mcs = self.mcs_data_container

        s = 0
        if search_uuid:
            # n is number of frames we need to send uuid
            n = int(ceil(32 / float(len(ceed_.count_indices)))) * 5

            # following searches for the uuid in the mcs data
            strides = mcs.count_data.strides + (mcs.count_data.strides[-1], )
            shape = mcs.count_data.shape[-1] - n + 1, n
            strided = np.lib.stride_tricks.as_strided(
                mcs.count_data, shape=shape, strides=strides)
            res = np.all(strided == ceed_.count_data[:n], axis=1)
            indices = np.mgrid[0:len(res)][res]

            if not len(indices):
                raise AlignmentException('Could not find alignment')
            if len(indices) > 1:
                raise Exception(
                    'Found multiple Ceed-mcs alignments ({})'.format(indices))

            s = indices[0]

        e = len(ceed_.clock_data) - 1 + s
        mcs_indices = mcs.data_indices_start[s:e]

        if np.all(mcs.clock_data[s:e] == ceed_.clock_data[:-1]) and \
            np.all(mcs.short_count_data[s:e] == ceed_.short_count_data[:-1]) \
                and np.all(mcs.count_data[s:e] == ceed_.count_data[:-1]):
            return mcs_indices

        raise AlignmentException('Could not align the data')

    def estimate_skipped_frames(self, ceed_mcs_alignment):
        from ceed.analysis import CeedDataReader
        return CeedDataReader.compute_long_and_skipped_frames(
            self.n_sub_frames, self.ceed_data['rendered_frames'],
            ceed_mcs_alignment)

    def get_skipped_frames_summary(self, ceed_mcs_alignment, experiment_num):
        mcs_long_frames, mcs_frame_len, ceed_skipped, ceed_skipped_main, \
            largest_bad = self.estimate_skipped_frames(ceed_mcs_alignment)
        num_long = sum(i - 1 for i in mcs_frame_len)
        main_skipped = len(ceed_skipped_main)
        skipped = len(ceed_skipped)

        return (
            f'Aligned experiment {experiment_num: >2}. '
            f'MCS: [{ceed_mcs_alignment[0]: 10} - '
            f'{ceed_mcs_alignment[-1]: 10}] '
            f'({len(ceed_mcs_alignment): 7} frames). {num_long: 3} slow, '
            f'{skipped: 3} ({main_skipped: 2}) dropped. Max {largest_bad} bad')

    def merge_data(
            self, filename, alignment_indices, notes='', notes_filename=None):
        if os.path.exists(filename):
            raise Exception('{} already exists'.format(filename))

        if notes_filename and os.path.exists(notes_filename):
            with open(notes_filename, 'r') as fh:
                lines = fh.read()

            if notes and lines:
                notes += '\n'
            notes += lines

        mcs_f = McsPy.McsData.RawData(self.mcs_filename)
        if not len(mcs_f.recordings):
            raise Exception('There is no data in {}'.format(self.mcs_filename))
        if len(mcs_f.recordings) > 1:
            raise Exception('There is more than one recording in {}'.
                            format(self.mcs_filename))

        copy2(self.ceed_filename, filename)
        f = nix.File.open(filename, nix.FileMode.ReadWrite)

        if 'app_logs' not in f.sections:
            f.create_section('app_logs', 'log')
            f.sections['app_logs']['notes'] = ''

        if f.sections['app_logs']['notes'] and notes:
            f.sections['app_logs']['notes'] += '\n'
        f.sections['app_logs']['notes'] += notes

        block = f.create_block(
            'ceed_mcs_alignment', 'each row, r, contains the sample index '
            'in the mcs data corresponding with row r in ceed data. This is '
            'the index at which the corresponding Ceed frame was displayed')
        for exp, indices in alignment_indices.items():
            block.create_data_array(
                'experiment_{}'.format(exp), 'experiment_{}'.format(exp),
                data=indices)

        streams = mcs_f.recordings[0].analog_streams
        num_channels = 0
        for stream in streams.values():
            if 128 in stream.channel_infos:
                num_channels += 1
            elif 0 in stream.channel_infos:
                num_channels += len(stream.channel_infos)

        pbar = tqdm(
            total=num_channels, file=sys.stdout, unit_scale=1,
            unit='data channels')

        block = f.create_block('mcs_data', 'mcs_raw_experiment_data')
        block.metadata = section = f.create_section(
            'mcs_metadata', 'Associated metadata for the mcs data')
        for stream_id, stream in streams.items():
            if 128 in stream.channel_infos:
                pbar.update()
                block.create_data_array(
                    'digital_io', 'digital_io',
                    data=np.array(stream.channel_data).squeeze())
            elif 0 in stream.channel_infos:
                for i in stream.channel_infos:
                    info = stream.channel_infos[i].info
                    pbar.update()

                    elec_sec = section.create_section(
                        'electrode_{}'.format(info['Label']),
                        'metadata for this electrode')
                    for key, val in info.items():
                        if isinstance(val, np.generic):
                            val = val.item()
                        elec_sec[key] = yaml_dumps(val)

                    freq = stream.channel_infos[i].sampling_frequency
                    freq = freq.m_as(ureg.hertz).item()
                    ts = stream.channel_infos[i].sampling_tick
                    ts = ts.m_as(ureg.second).item()

                    elec_sec['sampling_frequency'] = yaml_dumps(freq)
                    elec_sec['sampling_tick'] = yaml_dumps(ts)

                    data = np.array(stream.channel_data[i, :])
                    block.create_data_array(
                        'electrode_{}'.format(info['Label']), 'electrode_data',
                        data=data)

        pbar.close()
        f.close()


if __name__ == '__main__':
    ceed_file = r'/home/cpl/Desktop/data/3-11-19/slice2.h5'
    mcs_file = \
        r'/home/cpl/Desktop/data/3-11-19/2019-03-11T17-13-30McsRecording.h5'
    output_file = r'/home/cpl/Desktop/data/3-11-19/slice2_merged.h5'
    notes = ''
    notes_filename = None

    merger = CeedMCSDataMerger(ceed_filename=ceed_file, mcs_filename=mcs_file)

    merger.read_mcs_data()
    merger.read_ceed_data()
    merger.parse_mcs_data()

    alignment = {}
    for experiment in merger.get_experiment_numbers([]):
        merger.read_ceed_experiment_data(experiment)
        merger.parse_ceed_experiment_data()

        try:
            align = alignment[experiment] = merger.get_alignment()
            print(merger.get_skipped_frames_summary(align, experiment))
        except Exception as e:
            print(
                "Couldn't align MCS and ceed data for experiment "
                "{} ({})".format(experiment, e))

    merger.merge_data(
        output_file, alignment, notes=notes, notes_filename=notes_filename)
