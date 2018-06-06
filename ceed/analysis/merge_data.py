import datetime
import numbers
import os.path
from math import ceil
import re
from shutil import copy2
from McsPy import ureg
import McsPy.McsData
import numpy as np
import nixio as nix
from cplcom.utils import yaml_dumps, yaml_loads

McsPy.McsData.VERBOSE = False


class AlignmentException(Exception):
    pass


class CeedMCSDataMerger(object):

    ceed_config_orig = {}

    ceed_data = {}

    mcs_dig_data = []

    mcs_dig_config = {}

    mcs_mapping = {}

    ceed_mapping = {}

    _experiment_pat = re.compile('^experiment([0-9]+)$')

    @staticmethod
    def get_experiment_names(filename):
        nix_file = nix.File.open(filename, nix.FileMode.ReadOnly)
        experiments = []

        for block in nix_file.blocks:
            m = re.match(CeedMCSDataMerger._experiment_pat, block.name)
            if m is not None:
                experiments.append(m.group(1))

        nix_file.close()
        return list(sorted(experiments, key=int))

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
        f = nix.File.open(filename, nix.FileMode.ReadOnly)

        block = f.blocks['experiment{}'.format(experiment)]
        section = block.metadata
        start_t = datetime.datetime(1970, 1, 1) + \
            datetime.timedelta(seconds=block.created_at)

        metadata = {}
        try:
            try:
                config = section.sections['app_config']
            except KeyError:
                raise Exception('Did not find config in experiment info')

            for prop in config:
                metadata[prop.name] = yaml_loads(prop.values[0].value)

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

    def create_dig_mappings(self):
        config = self.ceed_config_orig['serializer']
        if 'config' in self.ceed_mapping and \
                config == self.ceed_mapping['config']:
            return

        ceed_mcs_map = config['projector_to_aquisition_map']

        ceed_map = self.ceed_mapping = {}
        mcs_map = self.mcs_mapping = {}
        indices = np.arange(2**16, dtype=np.uint16)

        ceed_short_count_indices = config['short_count_indices']
        ceed_short_counter_map_l = np.zeros((2 ** 16, ), dtype=np.uint16)
        ceed_short_counter_map_u = np.zeros((2 ** 16, ), dtype=np.uint16)
        mcs_short_counter_map = np.zeros((2 ** 16, ), dtype=np.uint16)

        for i, k in enumerate(ceed_short_count_indices):
            if k <= 15:
                ceed_short_counter_map_l |= ((indices & (1 << k)) >> k) << i
            else:
                ceed_short_counter_map_u |= (
                    (indices & (1 << (k % 16))) >> (k % 16) << i)
            k_mcs = ceed_mcs_map[k]
            mcs_short_counter_map |= (indices & (1 << k_mcs)) >> (k_mcs - i)
        ceed_map['short_count_l'] = ceed_short_counter_map_l
        ceed_map['short_count_u'] = ceed_short_counter_map_u
        mcs_map['short_count'] = mcs_short_counter_map

        ceed_count_indices = config['count_indices']
        ceed_counter_map_l = np.zeros((2 ** 16, ), dtype=np.uint16)
        ceed_counter_map_u = np.zeros((2 ** 16, ), dtype=np.uint16)
        mcs_counter_map = np.zeros((2 ** 16, ), dtype=np.uint16)

        for i, k in enumerate(ceed_count_indices):
            if k <= 15:
                ceed_counter_map_l |= ((indices & (1 << k)) >> k) << i
            else:
                ceed_counter_map_u |= (
                    (indices & (1 << (k % 16))) >> (k % 16) << i)
            k_mcs = ceed_mcs_map[k]
            mcs_counter_map |= (indices & (1 << k_mcs)) >> (k_mcs - i)
        ceed_map['count_l'] = ceed_counter_map_l
        ceed_map['count_u'] = ceed_counter_map_u
        mcs_map['count'] = mcs_counter_map

    def parse_digital_data(
            self, ceed_file, mcs_file, ceed_experiment, find_by='uuid'):
        self.read_mcs_digital_data(mcs_file)
        self.read_ceed_digital_data(ceed_file, ceed_experiment)
        self.create_dig_mappings()

        config = self.ceed_config_orig['serializer']
        ceed_mcs_map = config['projector_to_aquisition_map']
        ceed_clock_idx = config['clock_idx']
        ceed_clock_bit = 1 << config['clock_idx']
        mcs_clock_idx = ceed_mcs_map[config['clock_idx']]
        mcs_clock_bit = 1 << ceed_mcs_map[config['clock_idx']]

        mcs_dig = self.mcs_dig_data
        mcs_t_start = self.mcs_dig_config['t_start']
        mcs_f = self.mcs_dig_config['f']

        ceed_t_start = self.ceed_data['start_t']
        ceed_bits = self.ceed_data['frame_bits']

        ceed_map = self.ceed_mapping
        mcs_map = self.mcs_mapping

        mcs_offset = 0
        if find_by is not None and isinstance(find_by, numbers.Number):
            mcs_offset = (ceed_t_start - mcs_t_start).total_seconds() - \
                float(find_by)
            if mcs_offset < 0:
                raise Exception('Ceed data is not in the mcs data')
            mcs_offset = int(mcs_offset * mcs_f)
            mcs_dig = mcs_dig[mcs_offset:]

        if len(mcs_dig) < 10:
            raise Exception('There is not enough data in the mcs file')

        ceed_clock_data = (ceed_bits & ceed_clock_bit) >> ceed_clock_idx
        mcs_clock_data = (mcs_dig & mcs_clock_bit) >> mcs_clock_idx

        ceed_short_counter_data = (
                ceed_map['short_count_l'][ceed_bits & 0xFFFF] |
                ceed_map['short_count_u'][(ceed_bits & 0xFFFF0000) >> 16])
        mcs_short_counter_data = mcs_map['short_count'][mcs_dig]
        ceed_counter_bits_data = (
                ceed_map['count_l'][ceed_bits & 0xFFFF] |
                ceed_map['count_u'][(ceed_bits & 0xFFFF0000) >> 16])
        mcs_counter_bits_data = mcs_map['count'][mcs_dig]

        clock_change = np.argwhere(
            mcs_clock_data[1:] - mcs_clock_data[:-1]).squeeze()
        mcs_idx_start = np.array([0], dtype=clock_change.dtype)
        end = np.array([len(mcs_clock_data)], dtype=clock_change.dtype)
        if len(clock_change):
            mcs_idx_start = np.concatenate((mcs_idx_start, clock_change + 1))
            end = np.concatenate((clock_change, end))
        mcs_indices = (mcs_idx_start + end) // 2

        s = 0 if mcs_clock_data[0] else 1
        mcs_indices = mcs_indices[s:]
        mcs_idx_start = mcs_idx_start[s:] + mcs_offset
        mcs_clock_data = mcs_clock_data[mcs_indices]
        mcs_short_counter_data = mcs_short_counter_data[mcs_indices]
        mcs_counter_bits_data = mcs_counter_bits_data[mcs_indices]

        return (
            config, mcs_idx_start, mcs_clock_data,
            mcs_short_counter_data, mcs_counter_bits_data, ceed_clock_data,
            ceed_short_counter_data, ceed_counter_bits_data, find_by)

    def get_alignment(
            self, config, mcs_idx_start, mcs_clock_data,
            mcs_short_counter_data, mcs_counter_bits_data, ceed_clock_data,
            ceed_short_counter_data, ceed_counter_bits_data, find_by):

        # counter_bit_width = 32
        if find_by == 'uuid':  # find by uuid
            ceed_d = ceed_counter_bits_data
            mcs_d = mcs_counter_bits_data
            n = int(ceil(32 / float(len(config['count_indices'])))) * 5

            strides = mcs_d.strides + (mcs_d.strides[-1], )
            shape = mcs_d.shape[-1] - n + 1, n
            strided = np.lib.stride_tricks.as_strided(
                mcs_counter_bits_data, shape=shape, strides=strides)
            res = np.all(strided == ceed_d[:n], axis=1)
            indices = np.mgrid[0:len(res)][res]

            if not len(indices):
                raise AlignmentException('Could not find alignment')
            if len(indices) > 1:
                raise Exception(
                    'Found multiple Ceed-mcs alignment ({})'.format(indices))

            i = indices[0]
            mcs_idx_start = mcs_idx_start[i:]
            mcs_clock_data = mcs_clock_data[i:]

            mcs_short_counter_data = mcs_short_counter_data[i:]
            mcs_counter_bits_data = mcs_counter_bits_data[i:]

        ceed_clock_data = ceed_clock_data[:-1]
        ceed_short_counter_data = ceed_short_counter_data[:-1]
        ceed_counter_bits_data = ceed_counter_bits_data[:-1]

        mcs_idx_start = mcs_idx_start[:len(ceed_clock_data)]
        mcs_clock_data = mcs_clock_data[:len(ceed_clock_data)]
        mcs_short_counter_data = mcs_short_counter_data[:len(ceed_clock_data)]
        mcs_counter_bits_data = mcs_counter_bits_data[:len(ceed_clock_data)]

        if np.all(mcs_clock_data == ceed_clock_data) and\
                np.all(mcs_short_counter_data == ceed_short_counter_data) and \
                np.all(mcs_counter_bits_data == ceed_counter_bits_data):
            return mcs_idx_start

        raise AlignmentException('Could not align the data')

        e = 20
        # print(mcs_idx_start[:e])
        # print(mcs_clock_data[:e])
        # print(mcs_short_counter_data[:e])
        # print(mcs_counter_bits_data[:e])
        # print(ceed_clock_data[:e])
        # print(ceed_short_counter_data[:e])
        # print(ceed_counter_bits_data[:e])
        # print(len(ceed_short_counter_data), len(mcs_short_counter_data))

    def merge_data(
            self, filename, ceed_filename, mcs_filename, alignment_indices):
        if os.path.exists(filename):
            raise Exception('{} already exists'.format(filename))

        mcs_f = McsPy.McsData.RawData(mcs_filename)
        if not len(mcs_f.recordings):
            raise Exception('There is no data in {}'.format(mcs_filename))
        if len(mcs_f.recordings) > 1:
            raise Exception('There is more than one recording in {}'.
                            format(mcs_filename))

        copy2(ceed_filename, filename)
        f = nix.File.open(filename, nix.FileMode.ReadWrite)

        block = f.create_block(
            'ceed_mcs_alignment', 'each row, r, contains the sample number '
            'in the mcs data corresponding with row r in ceed data.')
        for exp, indices in alignment_indices.items():
            block.create_data_array(
                'experiment_{}'.format(exp), 'experiment_{}'.format(exp),
                data=indices)

        block = f.create_block('mcs_data', 'mcs_raw_experiment_data')
        block.metadata = section = f.create_section(
            'mcs_metadata', 'Associated metadata for the mcs data')
        for stream_id in mcs_f.recordings[0].analog_streams:
            stream = mcs_f.recordings[0].analog_streams[stream_id]
            if 128 in stream.channel_infos:
                block.create_data_array(
                    'digital_io', 'digital_io',
                    data=np.array(stream.channel_data).squeeze())
            elif 0 in stream.channel_infos:
                for i in stream.channel_infos:
                    print('writing channel {}'.format(i))
                    info = stream.channel_infos[i].info

                    elec_sec = section.create_section(
                        'electrode_{}'.format(info['Label']),
                        'metadata for this electrode')
                    for key, val in info.items():
                        #print(key, val, type(key), type(val))
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
        f.close()


if __name__ == '__main__':
    ceed_file = r'/home/cpl/Desktop/test.h5'
    mcs_file = r'/home/cpl/share/2018-04-13T11-43-20McsRecording.h5'
    output_file = r'/home/cpl/Desktop/test_out.h5'
    data = CeedMCSDataMerger()

    alignment = {}
    for experiment in data.get_experiment_names(ceed_file):
        vals = data.parse_digital_data(ceed_file, mcs_file, experiment)
        try:
            alignment[experiment] = data.get_alignment(*vals)
            print(experiment)
        except AlignmentException:
            print("Couldn't align {}".format(experiment))
        except Exception as e:
            print("{}: {}".format(e, experiment))
    data.merge_data(output_file, ceed_file, mcs_file, alignment)
