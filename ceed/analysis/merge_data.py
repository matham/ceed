"""Ceed-MCS data merging
=========================

The classes to merge and align the Ceed and MCS recording data into a single
file.
"""

import sys
import datetime
from tqdm import tqdm
import numbers
import os.path
from math import ceil
from typing import List, Dict, Optional
from shutil import copy2
from McsPy import ureg
import McsPy.McsData
import numpy as np
import nixio as nix
from more_kivy_app.utils import yaml_dumps, yaml_loads
from ceed.storage.controller import num_ticks_handshake, \
    num_ticks_handshake_1_0_0_dev0

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

    def get_count_ints(self, contains_sub_frames, n_sub_frames):
        """Counter with sub-frames removed.
        """
        n_counter_bits = len(self.count_indices)
        n_parts_per_int = self.n_parts_per_int
        # all sub frames are the same (if we have them)
        # We don't need inverted frames
        count = self.count_data
        if contains_sub_frames:
            count = self.count_data[::n_sub_frames]

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
            self, count_2d, count_inverted_2d, n_handshake_ints,
            contains_sub_frames, n_sub_frames):
        if count_2d.shape[0]:
            # even if no handshake sent, the first value will be size (zero)
            assert n_handshake_ints
        count_shape = count_2d.shape

        # set count bits
        count_bits = 0
        for i in range(len(self.count_indices)):
            count_bits |= 1 << i

        # config is not inverted
        if not np.all(count_2d[:n_handshake_ints, :]
                      == count_inverted_2d[:n_handshake_ints, :]):
            raise ValueError('Handshake data corrupted')

        # first item is not inverted
        if not np.all(count_2d[:, 0] == count_inverted_2d[:, 0]):
            raise ValueError('Non-inverted data group corrupted')

        count_2d = count_2d[n_handshake_ints:, :]
        count_inverted_2d = count_inverted_2d[n_handshake_ints:, :]
        if not np.all(
                (count_2d[:, 1:] ^ count_bits) & count_bits ==
                count_inverted_2d[:, 1:]):
            raise ValueError('Inverted data group corrupted')

        count_data = self.count_data
        if contains_sub_frames:
            count_data = count_data[::n_sub_frames]
        count_data = count_data[count_shape[0] * count_shape[1] * 2:]
        # we can't check the last item if it doesn't have inverted value
        if len(count_data) % 2:
            count_data = count_data[:-1]

        # n_handshake_ints >= 1, if we have at least one full int
        if not n_handshake_ints:
            # only have partial int, handshake is not inverted
            assert not count_2d.shape[0]
            if not np.all(count_data[::2] == count_data[1::2]):
                raise ValueError('Handshake length data corrupted')
        elif count_2d.shape[0] < n_handshake_ints:
            # have handshake size and remaining is handshake item, not inverted
            if not np.all(count_data[::2] == count_data[1::2]):
                raise ValueError('Remaining handshake data corrupted')
        else:
            # remaining is regular data, first val is not inverted, the rest is
            if not np.all(count_data[:1] == count_data[1:2]):
                raise ValueError('Remaining handshake data corrupted')
            if not np.all((count_data[2::2] ^ count_bits) & count_bits
                          == count_data[3::2]):
                raise ValueError('Remaining inverted handshake data corrupted')

    def get_handshake(
            self, count: np.ndarray, contains_sub_frames, n_sub_frames):
        """If handshake is incomplete returns empty.
        """
        if len(count):
            n_config = int(count[0])
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

    def check_missing_frames(self, contains_sub_frames, n_sub_frames):
        short = self.short_count_data
        if contains_sub_frames:
            short = short[::n_sub_frames]
        max_shot_val = 2 ** len(self.short_count_indices)

        i = short[0]
        for k, val in enumerate(short):
            if i != val:
                raise ValueError(f'Skipped a frame at frame {k}')
            i += 1
            if i == max_shot_val:
                i = 0

    def get_counter(self):
        n_frames = len(frame_bits)
        counter = np.zeros(n_frames, dtype=np.uint64)
        if not n_frames:
            return

        short = self.short_count_data
        max_shot_val = 2 ** len(self.short_count_indices)
        # frames on which the counter starts a new value
        counter_frames = np.zeros(n_frames, dtype=np.bool)
        counter_frames[n_config_frames::2 * self.n_parts_per_int] = True
        # last frame doesn't have complete value
        remainder = n_frames % (2 * self.n_parts_per_int)
        if remainder:
            counter_frames[-remainder] = False

        counter[counter_frames] = count[n_handshake_ints:]
        # first number is always 1
        last_count = counter[0] = 1

        last_short = short[0]
        for i in range(1, len(frame_bits)):
            if not counter_frames[i]:
                if short[i] >= last_short:
                    counter[i] = last_count + short[i] - last_short
                else:
                    # it overflowed to zero
                    counter[i] = last_count + \
                        short[i] + max_shot_val - last_short

            last_count = counter[i]
            last_short = short[i]


class MCSDigitalData(DigitalDataStore):

    data_indices_center: np.ndarray = None

    data_indices_start: np.ndarray = None

    data_indices_end: np.ndarray = None

    def parse(
            self, ceed_version, data, t_start, f, n_sub_frames,
            find_start_from_ceed_time=False, estimated_start: float = 0,
            pre_estimated_start: float = 0):
        self._parse_components(data)
        self.reduce_samples(
            t_start, f, n_sub_frames, find_start_from_ceed_time,
            estimated_start, pre_estimated_start)

        if ceed_version == '1.0.0.dev0':
            return

    def reduce_samples(
            self, t_start, f, n_sub_frames, find_start_from_ceed_time=False,
            estimated_start: float = 0, pre_estimated_start: float = 0):
        clock_data = self.clock_data
        short_count_data = self.short_count_data
        count_data = self.count_data

        offset = 0
        if find_start_from_ceed_time:
            offset = (estimated_start - t_start).total_seconds() - \
                float(pre_estimated_start)
            if offset < 0:
                raise Exception('Ceed data is not in the mcs data')

            offset = int(offset * f)
            clock_data = clock_data[offset:]
            short_count_data = short_count_data[offset:]
            count_data = count_data[offset:]

        # should have at least 10 samples. At 5k sampling rate it's reasonable
        if len(clock_data) < 10:
            raise Exception('There is not enough data in the mcs file')

        clock_change = np.argwhere(clock_data[1:] - clock_data[:-1]).squeeze()
        # indices in data where value is different from last (including 0)
        idx_start = np.array([0], dtype=clock_change.dtype)
        # indices in data where next value is different (including last value)
        idx_end = np.array([len(clock_data) - 1], dtype=clock_change.dtype)
        if len(clock_change):
            idx_start = np.concatenate((idx_start, clock_change + 1))
            idx_end = np.concatenate((clock_change, idx_end))

        # center index in data of the clock  high or low region
        indices = (idx_start + idx_end) // 2

        # start at the
        s = 0 if clock_data[0] else 1
        indices = indices[s:]

        # indices in the original data
        self.data_indices_start = idx_start[s:] + offset
        self.data_indices_center = indices + offset
        self.data_indices_end = idx_end[s:] + offset
        # condensed data
        self.clock_data = clock_data[indices]
        self.short_count_data = short_count_data[indices]
        self.count_data = count_data[indices]


class CeedDigitalData(DigitalDataStore):

    def parse(
            self, ceed_version, frame_bits: np.ndarray,
            frame_counter: np.ndarray, start_t: np.ndarray, n_sub_frames: int
    ) -> None:
        if ceed_version == '1.0.0.dev0':
            self.parse_data_v1_0_0_dev0(frame_bits)
        else:
            self.parse_data(frame_bits, frame_counter, start_t, n_sub_frames)

    def parse_data_v1_0_0_dev0(self, data: np.ndarray) -> None:
        self._parse_components(data)

    def parse_data(
            self, frame_bits: np.ndarray, frame_counter: np.ndarray,
            start_t: np.ndarray, n_sub_frames: int) -> None:
        # the last frame(s) may or may not have been rendered (e.g. with
        # sub-frames, all the sub-frames may not have been rendered)
        contains_sub_frames = True
        self._parse_components(frame_bits)

        # use short counter to see if missing frames
        self.check_missing_frames(contains_sub_frames, n_sub_frames)
        # chop of partial ints and get full ints
        count, count_2d, count_inverted_2d = self.get_count_ints(
            contains_sub_frames, n_sub_frames)
        # get handshake from full ints
        handshake_data, handshake_len, n_handshake_ints, n_config_frames = \
            self.get_handshake(count, contains_sub_frames, n_sub_frames)
        # check that full and partial ints match
        self.check_counter_consistency(
            count_2d, count_inverted_2d, n_handshake_ints, contains_sub_frames,
            n_sub_frames)

        self.handshake_data = handshake_data
        self.expected_handshake_len = handshake_len
        self.counter = frame_counter.copy()


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
        video_mode = self.ceed_global_config['view']['video_mode']
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
            except KeyError:
                raise Exception(
                    'Did not find config in experiment info for experiment {}'.
                    format(experiment))

            for prop in config:
                metadata[prop.name] = yaml_loads(read_nix_prop(prop))

            if not block.data_arrays['frame_bits'].shape or \
                    not block.data_arrays['frame_bits'].shape[0]:
                raise Exception('Experiment {} has no data'.format(experiment))

            frame_bits = np.asarray(block.data_arrays['frame_bits']).squeeze()
            frame_counter = np.asarray(
                block.data_arrays['frame_counter']).squeeze()

            rendered_counter = np.asarray(
                block.data_arrays['frame_time_counter']).squeeze()
            if not np.all(
                    rendered_counter[1:] - rendered_counter[:-1]
                    == self.n_sub_frames):
                raise ValueError('Some frames were not rendered and skipped')

        except Exception:
            f.close()
            raise
        else:
            f.close()

        self.ceed_data = {
            'frame_bits': frame_bits, 'frame_counter': frame_counter,
            'start_t': start_t}

        self.ceed_config_orig = metadata['app_settings']

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

        n_sub_frames = self.n_sub_frames

        self.create_or_reuse_mcs_data_container()
        self.mcs_data_container.parse(
            self.ceed_version, self.mcs_dig_data,
            self.mcs_dig_config['t_start'], self.mcs_dig_config['f'],
            n_sub_frames, find_start_from_ceed_time=find_start_from_ceed_time,
            pre_estimated_start=pre_estimated_start,
            estimated_start=estimated_start)

    def get_alignment(self, search_uuid=True):
        if self.ceed_version == '1.0.0.dev0':
            return self._get_alignment_v1_0_0_dev0(search_uuid)

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

    @staticmethod
    def merge_data(
            filename, ceed_filename, mcs_filename, alignment_indices,
            notes='', notes_filename=None):
        if os.path.exists(filename):
            raise Exception('{} already exists'.format(filename))

        if notes_filename and os.path.exists(notes_filename):
            with open(notes_filename, 'r') as fh:
                lines = fh.read()

            if notes and lines:
                notes += '\n'
            notes += lines

        mcs_f = McsPy.McsData.RawData(mcs_filename)
        if not len(mcs_f.recordings):
            raise Exception('There is no data in {}'.format(mcs_filename))
        if len(mcs_f.recordings) > 1:
            raise Exception('There is more than one recording in {}'.
                            format(mcs_filename))

        copy2(ceed_filename, filename)
        f = nix.File.open(filename, nix.FileMode.ReadWrite)

        if 'app_logs' not in f.sections:
            f.create_section('app_logs', 'log')
            f.sections['app_logs']['notes'] = ''

        if f.sections['app_logs']['notes'] and notes:
            f.sections['app_logs']['notes'] += '\n'
        f.sections['app_logs']['notes'] += notes

        block = f.create_block(
            'ceed_mcs_alignment', 'each row, r, contains the sample number '
            'in the mcs data corresponding with row r in ceed data.')
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
            print(
                'Aligned MCS and ceed data for experiment {} at MCS samples '
                '[{} - {}] ({} frames)'.format(
                    experiment, align[0], align[-1], len(align)))
        except Exception as e:
            print(
                "Couldn't align MCS and ceed data for experiment "
                "{} ({})".format(experiment, e))

    merger.merge_data(
        output_file, ceed_file, mcs_file, alignment, notes=notes,
        notes_filename=notes_filename)
