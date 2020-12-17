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
import re
from shutil import copy2
from McsPy import ureg
import McsPy.McsData
import numpy as np
import nixio as nix
from more_kivy_app.utils import yaml_dumps, yaml_loads
from ceed.storage.controller import num_ticks_handshake

__all__ = ('CeedMCSDataMerger', 'DigitalDataStore', 'MCSDigitalData',
           'CeedDigitalData', 'AlignmentException')

McsPy.McsData.VERBOSE = False


class AlignmentException(Exception):
    pass


class DigitalDataStore(object):

    counter_bit_width = 32

    short_count_indices = None

    short_count_data = None

    count_indices = None

    count_data = None

    clock_index = None

    clock_data = None

    data = None
    '''raw data.

    maps the values from data at indices_center using the maps at xxx_map
    for each array item and saves it into xxx_data. xxx_map is formed from
    xxx_indices.

    data_indices_xxx are the indices in data where each xxx_data array item
    came from.
    '''

    def __init__(self, short_count_indices, count_indices, clock_index,
                 counter_bit_width):
        '''indices are first mapped with projector_to_aquisition_map. '''
        super(DigitalDataStore, self).__init__()
        self.short_count_indices = short_count_indices
        self.count_indices = count_indices
        self.clock_index = clock_index
        self.counter_bit_width = counter_bit_width
        self.populate_indices()

    def populate_indices(self):
        pass

    def compare_indices(self, short_count_indices, count_indices, clock_index):
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
        if not isinstance(other, MCSDigitalData):
            return False

        return self.compare_indices(
            other.short_count_indices, other.count_indices, other.clock_index)


class MCSDigitalData(DigitalDataStore):

    short_count_map = None

    count_map = None

    data_indices_center = None

    data_indices_start = None

    data_indices_end = None

    def populate_indices(self):
        indices = np.arange(2**16, dtype=np.uint16)

        short_count_map = np.zeros((2 ** 16, ), dtype=np.uint16)
        for i, k in enumerate(self.short_count_indices):
            short_count_map |= (indices & (1 << k)) >> (k - i)
        self.short_count_map = short_count_map

        count_map = np.zeros((2 ** 16, ), dtype=np.uint16)
        for i, k in enumerate(self.count_indices):
            count_map |= (indices & (1 << k)) >> (k - i)
        self.count_map = count_map

    def parse_data(
            self, data, t_start, f, find_offset=None, estimated_start=None):
        self.data = data
        clock_index = self.clock_index
        clock_bit = 1 << clock_index

        offset = 0
        if isinstance(find_offset, numbers.Number):
            offset = (estimated_start - t_start).total_seconds() - \
                float(find_offset)
            if offset < 0:
                raise Exception('Ceed data is not in the mcs data')

            offset = int(offset * f)
            data = data[offset:]

        if len(data) < 10:
            raise Exception('There is not enough data in the mcs file')

        clock_data = (data & clock_bit) >> clock_index
        short_count_data = self.short_count_map[data]
        count_data = self.count_map[data]

        clock_change = np.argwhere(clock_data[1:] - clock_data[:-1]).squeeze()
        idx_start = np.array([0], dtype=clock_change.dtype)
        idx_end = np.array([len(clock_data)], dtype=clock_change.dtype)
        if len(clock_change):
            idx_start = np.concatenate((idx_start, clock_change + 1))
            idx_end = np.concatenate((clock_change, idx_end))

        indices = (idx_start + idx_end) // 2
        s = 0 if clock_data[0] else 1
        indices = indices[s:]

        self.data_indices_start = idx_start[s:] + offset
        self.data_indices_center = indices + offset
        self.data_indices_end = idx_end[s:] + offset
        self.clock_data = clock_data[indices]
        self.short_count_data = short_count_data[indices]
        self.count_data = count_data[indices]


class CeedDigitalData(DigitalDataStore):

    short_count_l_map = None

    short_count_u_map = None

    count_l_map = None

    count_u_map = None

    def populate_indices(self):
        indices = np.arange(2 ** 16, dtype=np.uint16)

        short_map_l = np.zeros((2 ** 16, ), dtype=np.uint16)
        short_map_u = np.zeros((2 ** 16, ), dtype=np.uint16)

        for i, k in enumerate(self.short_count_indices):
            if k <= 15:
                short_map_l |= ((indices & (1 << k)) >> k) << i
            else:
                short_map_u |= (indices & (1 << (k % 16))) >> (k % 16) << i

        self.short_count_l_map = short_map_l
        self.short_count_u_map = short_map_u

        counter_map_l = np.zeros((2 ** 16, ), dtype=np.uint16)
        counter_map_u = np.zeros((2 ** 16, ), dtype=np.uint16)

        for i, k in enumerate(self.count_indices):
            if k <= 15:
                counter_map_l |= ((indices & (1 << k)) >> k) << i
            else:
                counter_map_u |= (indices & (1 << (k % 16))) >> (k % 16) << i

        self.count_l_map = counter_map_l
        self.count_u_map = counter_map_u

    def parse_data(self, data):
        self.data = data
        clock_index = self.clock_index
        clock_bit = 1 << clock_index

        self.clock_data = (data & clock_bit) >> clock_index
        self.short_count_data = (
            self.short_count_l_map[data & 0xFFFF] |
            self.short_count_u_map[(data & 0xFFFF0000) >> 16])
        self.count_data = (
            self.count_l_map[data & 0xFFFF] |
            self.count_u_map[(data & 0xFFFF0000) >> 16])


class CeedMCSDataMerger(object):

    ceed_config_orig = {}

    ceed_data = {}

    ceed_data_container = None

    mcs_dig_data = []

    mcs_dig_config = {}

    mcs_data_container = None

    @staticmethod
    def get_experiment_numbers(filename, ignore_list=None):
        from ceed.storage.controller import CeedDataWriterBase
        nix_file = nix.File.open(filename, nix.FileMode.ReadOnly)
        try:
            names = CeedDataWriterBase.get_blocks_experiment_numbers(
                nix_file.blocks, ignore_list)
        finally:
            nix_file.close()
        return names

    def read_mcs_digital_data(self, filename):
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

    def read_ceed_digital_data(self, filename, experiment):
        from ceed.storage.controller import CeedDataWriterBase
        from ceed.analysis import read_nix_prop
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

            frame_bits = np.array(block.data_arrays['frame_bits']).squeeze()
            frame_counter = np.array(block.data_arrays['frame_counter']
                                     ).squeeze()
        except Exception:
            f.close()
            raise
        else:
            f.close()
        self.ceed_data = {
            'frame_bits': frame_bits, 'frame_counter': frame_counter,
            'start_t': start_t}
        self.ceed_config_orig = metadata['app_settings']

    @staticmethod
    def _clean_array_repeat(data):
        if not len(data):
            return np.array([], dtype=np.uint64), np.array([], dtype=data.dtype)
        data2 = data.astype(np.int32)
        diff = data2[1:] - data2[:-1]
        indices = np.concatenate((np.array([True], dtype=np.bool), diff != 0))
        return np.arange(len(data))[indices], data[indices]

    @staticmethod
    def _stitch_bits_in_array(data, bits_per_item, even_only=True):
        if even_only:
            data = data[::2]
        data = data.astype(np.uint32)

        n_items_per_val = int(ceil(32 / float(bits_per_item)))
        n = len(data) // n_items_per_val
        data = data[:n * n_items_per_val]
        values = np.zeros((n, ), dtype=np.uint32)
        for i in range(n_items_per_val):
            values |= data[i::n_items_per_val] << (i * bits_per_item)
        return values

    def create_or_reuse_ceed_data_container(self):
        config = self.ceed_config_orig['serializer']
        short_count_indices = config['short_count_indices']
        count_indices = config['count_indices']
        clock_index = config['clock_idx']
        counter_bit_width = config['counter_bit_width']

        if self.ceed_data_container is None or not self.ceed_data_container.\
            compare_indices(
                short_count_indices, count_indices, clock_index):
            self.ceed_data_container = CeedDigitalData(
                short_count_indices, count_indices, clock_index,
                counter_bit_width)
        return self.ceed_data_container

    def create_or_reuse_mcs_data_container(self):
        config = self.ceed_config_orig['serializer']
        ceed_mcs_map = config['projector_to_aquisition_map']
        map_f = np.vectorize(lambda x: ceed_mcs_map[x])

        short_count_indices = map_f(config['short_count_indices'])
        count_indices = map_f(config['count_indices'])
        clock_index = map_f(config['clock_idx'])
        counter_bit_width = config['counter_bit_width']

        if self.mcs_data_container is None or not self.mcs_data_container.\
            compare_indices(
                short_count_indices, count_indices, clock_index):
            self.mcs_data_container = MCSDigitalData(
                short_count_indices, count_indices, clock_index,
                counter_bit_width)
        return self.mcs_data_container

    def parse_ceed_digital_data(self):
        self.create_or_reuse_ceed_data_container()
        self.ceed_data_container.parse_data(self.ceed_data['frame_bits'])

    def parse_mcs_digital_data(self, find_by=None):
        self.create_or_reuse_mcs_data_container()
        self.mcs_data_container.parse_data(
            self.mcs_dig_data, self.mcs_dig_config['t_start'],
            self.mcs_dig_config['f'],
            find_offset=find_by, estimated_start=self.ceed_data['start_t'])

    def get_alignment(self, find_by=None, force=False):
        ceed_ = self.ceed_data_container
        mcs = self.mcs_data_container

        s = 0
        counter_bit_width = ceed_.counter_bit_width
        if find_by is None:  # find by uuid
            # n is number of frames we need to send uuid
            n = num_ticks_handshake(counter_bit_width, ceed_.count_indices, 16)
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

        if force:
            print(min(mcs_indices), max(mcs_indices))
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

    align_by = None
    merger = CeedMCSDataMerger()

    merger.read_mcs_digital_data(mcs_file)
    init = False

    alignment = {}
    for experiment in merger.get_experiment_numbers(ceed_file, ignore_list=[]):
        merger.read_ceed_digital_data(ceed_file, experiment)
        merger.parse_ceed_digital_data()

        if not init or align_by is not None:
            merger.parse_mcs_digital_data(find_by=align_by)
            init = True

        try:
            align = alignment[experiment] = merger.get_alignment(
                find_by=align_by)
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
