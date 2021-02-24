'''Storage Controller
=======================

Handles all data aspects, from the storage, loading and saving of configuration
data to the acquisition and creation of experimental data.

'''
import nixio as nix
from os.path import exists, basename, splitext, split, join, isdir, dirname
from os import remove
import os
from tempfile import NamedTemporaryFile
from shutil import copy2
from math import ceil
from threading import Thread, RLock
from queue import Queue, Empty
import numpy as np
from functools import partial
import struct
from typing import Generator, Dict, List
import re
import time
from tree_config import get_config_children_names

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, ListProperty, \
    DictProperty, BooleanProperty, ObjectProperty
from kivy.app import App
from kivy.clock import Clock
from kivy.logger import Logger

from more_kivy_app.app import app_error
from more_kivy_app.utils import yaml_dumps, yaml_loads

__all__ = (
    'CeedDataWriterBase', 'DataSerializerBase', 'num_ticks_handshake',
    'num_ticks_handshake_1_0_0_dev0')


class CeedDataWriterBase(EventDispatcher):

    _config_props_ = ('root_path', 'backup_interval', 'compression')

    root_path = StringProperty('')

    backup_interval = NumericProperty(5.)

    filename = StringProperty('')

    read_only_file = BooleanProperty(False)

    backup_filename = ''

    nix_file = None

    compression = StringProperty('Auto')

    has_unsaved = BooleanProperty(False)

    config_changed = BooleanProperty(False)

    data_queue = None

    data_thread = None

    data_lock = None

    func_display_callback = None

    stage_display_callback = None

    shape_display_callback = None

    clear_all_callback = None

    _experiment_pat = re.compile('^experiment([0-9]+)$')

    _mea_trans_pat = re.compile(
        '.+(mea_transform:[ \n\\-\\[\\],0-9.]+\\]?\\n).+', flags=re.DOTALL)

    backup_event = None

    __events__ = ('on_experiment_change', )

    def __init__(self, **kwargs):
        super(CeedDataWriterBase, self).__init__(**kwargs)
        if (not os.environ.get('KIVY_DOC_INCLUDE', None) and
                self.backup_interval):
            self.backup_event = Clock.schedule_interval(
                partial(self.write_changes, scheduled=True),
                self.backup_interval)

    @property
    def nix_compression(self):
        if self.compression == 'ZIP':
            return nix.Compression.DeflateNormal
        elif self.compression == 'None':
            return nix.Compression.No
        return nix.Compression.Auto

    def get_function_plugin_contents(self):
        app = App.get_running_app()
        return app.function_factory.plugin_sources

    def get_stage_plugin_contents(self):
        app = App.get_running_app()
        return app.stage_factory.plugin_sources

    def gather_config_data_dict(self, stages_only=False):
        app = App.get_running_app()
        data = {}
        data['shape'] = app.shape_factory.get_state()
        data['function'] = app.function_factory.save_functions()
        data['stage'] = app.stage_factory.save_stages()
        if stages_only:
            return data

        App.get_running_app().dump_app_settings_to_file()
        App.get_running_app().load_app_settings_from_file()
        data['app_settings'] = App.get_running_app().app_settings
        return data

    def clear_existing_config_data(self):
        if self.clear_all_callback is not None:
            self.clear_all_callback()

        app = App.get_running_app()
        app.stage_factory.clear_stages(force=True)
        app.shape_factory.remove_all_groups()
        app.shape_factory.delete_all_shapes(keep_locked_shapes=False)
        app.function_factory.clear_added_funcs(force=True)

    def apply_config_data_dict(
            self, data, stages_only=False, requires_app_settings=True):
        app = App.get_running_app()

        if not stages_only and (
                requires_app_settings or 'app_settings' in data):
            try:
                app_settings = data['app_settings']
            except KeyError as e:
                raise KeyError(
                    'You attempted to import configuration data for the app '
                    'and stages/functions, but there is no app config in the '
                    'file. Did you mean to import only the stages?') from e
            # filter classes that are not of this app
            classes = get_config_children_names(app)
            app.app_settings = {cls: app_settings[cls] for cls in classes}
            app.apply_app_settings()

        from ceed.analysis import CeedDataReader
        funcs, stages = CeedDataReader.populate_config(
            data, app.shape_factory, app.function_factory, app.stage_factory)

        if self.func_display_callback:
            for func in funcs:
                self.func_display_callback(func)
        if self.stage_display_callback:
            for stage in stages:
                self.stage_display_callback(stage)

    def on_experiment_change(self, name, value):
        pass

    def get_filebrowser_callback(
            self, func, clear_data=False, ext=None, **kwargs):

        def callback(paths):
            if not paths:
                return
            fname = paths[0]
            if ext and not fname.endswith(ext):
                fname += ext

            self.root_path = dirname(fname)

            def discard_callback(discard):
                if clear_data and not discard:
                    return

                if clear_data:
                    self.close_file(force_remove_autosave=True)
                    self.clear_existing_config_data()
                func(fname, **kwargs)

            if clear_data and (self.has_unsaved or self.config_changed):
                yesno = App.get_running_app().yesno_prompt
                yesno.msg = 'There are unsaved changes.\nDiscard them?'
                yesno.callback = discard_callback
                yesno.open()
            else:
                discard_callback(True)
        return callback

    def ui_close(self, app_close=False):
        '''The UI asked for to close a file. We create a new one if the app
        doesn't close.

        If unsaved, will prompt if want to save
        '''
        if self.has_unsaved or self.config_changed:
            def close_callback(discard):
                if discard:
                    self.close_file(force_remove_autosave=True)
                    self.clear_existing_config_data()
                    if app_close:
                        App.get_running_app().stop()
                    else:
                        self.create_file('')

            yesno = App.get_running_app().yesno_prompt
            yesno.msg = ('There are unsaved changes.\n'
                         'Are you sure you want to discard them?')
            yesno.callback = close_callback
            yesno.open()
            return False
        else:
            self.close_file()
            self.clear_existing_config_data()
            if not app_close:
                self.create_file('')
            return True

    def create_file(self, filename, overwrite=False):
        if exists(filename) and not overwrite:
            raise ValueError('{} already exists'.format(filename))
        self.close_file()

        self.filename = filename

        if filename:
            head, tail = split(filename)
            name, ext = splitext(tail)
        else:
            if not isdir(self.root_path):
                self.root_path = os.path.expanduser('~')
            head = self.root_path
            ext = '.h5'
            name = 'default'
        temp = NamedTemporaryFile(
            suffix=ext, prefix=name + '_', dir=head, delete=False)
        self.backup_filename = temp.name
        temp.close()

        f = self.nix_file = nix.File.open(
            self.backup_filename, nix.FileMode.Overwrite,
            compression=self.nix_compression)
        self.config_changed = self.has_unsaved = True
        Logger.debug(
            'Ceed Controller (storage): Created tempfile {}, with file "{}"'.
            format(self.backup_filename, self.filename))

        f.create_section('app_config', 'configuration')
        f.create_section('app_logs', 'log')
        f.sections['app_logs']['log_data'] = ''
        f.sections['app_logs']['notes'] = ''

        f.create_section('function_plugin_sources', 'files')
        f.sections['function_plugin_sources']['contents'] = yaml_dumps(
            self.get_function_plugin_contents())

        f.create_section('stage_plugin_sources', 'files')
        f.sections['stage_plugin_sources']['contents'] = yaml_dumps(
            self.get_stage_plugin_contents())

        block = f.create_block('fluorescent_images', 'image')
        sec = f.create_section('fluorescent_images_metadata', 'metadata')
        block.metadata = sec
        self.save()

        self.dispatch('on_experiment_change', 'open', None)

    @staticmethod
    def upgrade_file(nix_file):
        import ceed
        version = yaml_dumps(ceed.__version__)

        if 'app_logs' not in nix_file.sections:
            nix_file.create_section('app_logs', 'log')
            nix_file.sections['app_logs']['log_data'] = ''
            nix_file.sections['app_logs']['notes'] = ''

        if 'function_plugin_sources' not in nix_file.sections:
            nix_file.create_section('function_plugin_sources', 'files')
            nix_file.sections['function_plugin_sources']['contents'] = \
                yaml_dumps({})

        if 'stage_plugin_sources' not in nix_file.sections:
            nix_file.create_section('stage_plugin_sources', 'files')
            nix_file.sections['stage_plugin_sources']['contents'] = \
                yaml_dumps({})

        if 'fluorescent_images' not in nix_file.blocks:
            block = nix_file.create_block('fluorescent_images', 'image')
            sec = nix_file.create_section(
                'fluorescent_images_metadata', 'metadata')
            block.metadata = sec

        for num in CeedDataWriterBase.get_blocks_experiment_numbers(
                nix_file.blocks):
            name = CeedDataWriterBase.get_experiment_block_name(num)
            metadata = nix_file.blocks[name].metadata

            if 'notes' not in metadata:
                metadata['notes'] = ''
            if 'save_time' not in metadata:
                metadata['save_time'] = '0'

            if 'ceed_version' not in metadata.sections['app_config']:
                metadata.sections['app_config']['ceed_version'] = version

        if 'ceed_version' not in nix_file.sections['app_config']:
            import ceed
            nix_file.sections['app_config']['ceed_version'] = version

    def open_file(self, filename, read_only=False):
        '''Loads the file's config and opens the file for usage. '''
        self.close_file()

        self.filename = filename
        self.read_only_file = read_only

        head, tail = split(filename)
        name, ext = splitext(tail)
        temp = NamedTemporaryFile(
            suffix=ext, prefix=name + '_', dir=head, delete=False)
        self.backup_filename = temp.name
        temp.close()

        copy2(filename, self.backup_filename)
        self.clear_existing_config_data()
        self.import_file(self.backup_filename)

        self.nix_file = nix.File.open(
            self.backup_filename, nix.FileMode.ReadWrite,
            compression=self.nix_compression)
        self.config_changed = self.has_unsaved = True
        Logger.debug(
            'Ceed Controller (storage): Created tempfile {}, from existing '
            'file "{}"'.format(self.backup_filename, self.filename))

        self.upgrade_file(self.nix_file)
        self.write_config()

        self.dispatch('on_experiment_change', 'open', None)

    def close_file(self, force_remove_autosave=False):
        '''Closes without saving the data. But if data was unsaved, it leaves
        the backup file unchanged.
        '''
        if self.data_thread:
            raise TypeError("Cannot close data during an experiment")
        if self.nix_file:
            self.nix_file.close()
            self.nix_file = None

        if (not self.has_unsaved and not self.config_changed or
                force_remove_autosave) and self.backup_filename:
            remove(self.backup_filename)

        Logger.debug(
            'Ceed Controller (storage): Closed tempfile {}, with '
            '"{}"'.format(self.backup_filename, self.filename))
        self.filename = self.backup_filename = ''
        self.config_changed = self.has_unsaved = False
        self.read_only_file = False

        self.dispatch('on_experiment_change', 'close', None)

    def import_file(self, filename, stages_only=False):
        '''Loads the file's config data. '''
        from ceed.analysis import read_nix_prop
        Logger.debug(
            'Ceed Controller (storage): Importing "{}"'.format(self.filename))
        data = {}
        f = nix.File.open(filename, nix.FileMode.ReadOnly)

        try:
            for prop in f.sections['app_config']:
                data[prop.name] = yaml_loads(read_nix_prop(prop))
        finally:
            f.close()

        self.apply_config_data_dict(data, stages_only=stages_only)
        self.config_changed = True

    def discard_file(self,):
        if not self.has_unsaved and not self.config_changed:
            return

        f = self.filename
        self.close_file(force_remove_autosave=True)
        self.clear_existing_config_data()
        if f:
            self.open_file(f, read_only=self.read_only_file)
        else:
            self.create_file('')

    def save_as(self, filename, overwrite=False):
        if exists(filename) and not overwrite:
            raise ValueError('{} already exists'.format(filename))
        self.save(filename, True)
        self.open_file(filename)
        self.save()

    def save(self, filename=None, force=False):
        '''Saves the changes to the autosave and also saves the changes to
        the file in filename (if None saves to the current filename).
        '''
        if self.read_only_file and not force:
            raise TypeError('Cannot save because file was opened as read only. '
                            'Try saving as a new file')

        if not force and not self.has_unsaved and not self.config_changed:
            return

        try:
            if self.data_lock:
                self.data_lock.acquire()

            self.write_changes()
            filename = filename or self.filename
            if filename:
                copy2(self.backup_filename, filename)
        finally:
            if self.data_lock:
                self.data_lock.release()

        if filename:
            self.config_changed = self.has_unsaved = False

    def write_changes(self, *largs, scheduled=False):
        '''Writes unsaved changes to the current (autosave) file. '''
        if not self.nix_file or scheduled and self.read_only_file:
            return

        try:
            if self.data_lock:
                self.data_lock.acquire()

            if self.config_changed:
                self.write_config()
                self.dispatch('on_experiment_change', 'app_config', None)
            self.nix_file._h5file.flush()
        finally:
            if self.data_lock:
                self.data_lock.release()

    def write_yaml_config(self, filename, overwrite=False, stages_only=False):
        if exists(filename) and not overwrite:
            raise ValueError('{} already exists'.format(filename))

        data = yaml_dumps(self.gather_config_data_dict(stages_only=stages_only))
        with open(filename, 'w') as fh:
            fh.write(data)

    def read_yaml_config(
            self, filename, stages_only=False, requires_app_settings=True):
        with open(filename, 'r') as fh:
            data = fh.read()
        data = yaml_loads(data)
        self.apply_config_data_dict(
            data, stages_only=stages_only,
            requires_app_settings=requires_app_settings)
        self.config_changed = True

    def write_config(self, config_section=None):
        import ceed
        config = config_section if config_section is not None else \
            self.nix_file.sections['app_config']
        data = self.gather_config_data_dict()
        for k, v in data.items():
            config[k] = yaml_dumps(v)

        config['ceed_version'] = yaml_dumps(ceed.__version__)
        self.has_unsaved = True

    def read_config(self, config_section=None):
        from ceed.analysis import read_nix_prop
        config = config_section if config_section is not None else \
            self.nix_file.sections['app_config']
        data = {}
        for prop in config.props:
            data[prop.name] = yaml_loads(read_nix_prop(prop))

        return data

    def add_log_item(self, text):
        section = self.nix_file.sections['app_logs']
        t = time.time()
        section['log_data'] += '\n{}: {}'.format(t, text)
        self.has_unsaved = True
        self.dispatch('on_experiment_change', 'app_log', None)

    def add_app_log(self, text):
        app = App.get_running_app()
        if app is None:
            return

        app.error_indicator.add_item(text, 'user')

    def get_log_data(self):
        return self.nix_file.sections['app_logs']['log_data']

    def load_last_fluorescent_image(self, filename):
        from ceed.analysis import CeedDataReader
        f = nix.File.open(filename, nix.FileMode.ReadOnly)
        Logger.debug(
            'Ceed Controller (storage): Importing fluorescent image from '
            '"{}"'.format(self.filename))

        try:
            if not f.blocks:
                raise ValueError('Image not found in {}'.format(filename))

            names = set(
                self.get_experiment_block_name(i) for i in
                CeedDataWriterBase.get_blocks_experiment_numbers(f.blocks))
            for block in reversed(f.blocks):
                if block.name not in names:
                    continue

                img = CeedDataReader.read_image_from_block(block)
                if img is not None:
                    return img

            raise ValueError('Image not found in {}'.format(filename))
        finally:
            f.close()

    def add_image_to_file(self, img, notes=''):
        block = self.nix_file.blocks['fluorescent_images']
        n = len(block.groups)
        group = self.write_fluorescent_image(block, img, '_{}'.format(n))

        group.metadata['notes'] = notes
        group.metadata['save_time'] = '{}'.format(time.time())

        self.has_unsaved = True
        self.dispatch('on_experiment_change', 'image_add', n)

    def write_fluorescent_image(self, block, img, postfix=''):
        group = block.create_group(
            'fluorescent_image{}'.format(postfix), 'image')

        config = block.metadata.create_section(
            'fluorescent_image_config{}'.format(postfix), 'image')
        config['size'] = yaml_dumps(img.get_size())
        config['pix_fmt'] = img.get_pixel_format()
        config['linesizes'] = yaml_dumps(img.get_linesizes())
        config['buffer_size'] = yaml_dumps(img.get_buffer_size())
        group.metadata = config

        for i, plane in enumerate(img.to_bytearray()):
            image_data = block.create_data_array(
                'fluorescent_image_plane{}_{}'.format(postfix, i), 'image',
                dtype=np.uint8, data=plane)
            group.data_arrays.append(image_data)
        return group

    def get_num_fluorescent_images(self):
        return self.get_file_num_fluorescent_images(self.nix_file)

    @staticmethod
    def get_file_num_fluorescent_images(nix_file):
        try:
            return len(nix_file.blocks['fluorescent_images'].groups)
        except KeyError:
            return 0

    def get_experiment_numbers(self):
        return self.get_blocks_experiment_numbers(self.nix_file.blocks)

    def get_experiment_notes(self, experiment_block_number):
        name = self.get_experiment_block_name(experiment_block_number)
        if 'notes' in self.nix_file.blocks[name].metadata:
            return self.nix_file.blocks[name].metadata['notes']
        return ''

    def set_experiment_notes(self, experiment_block_number, text):
        name = self.get_experiment_block_name(experiment_block_number)
        block = self.nix_file.blocks[name]
        block.metadata['notes'] = text
        self.has_unsaved = True

        self.dispatch(
            'on_experiment_change', 'experiment_notes', experiment_block_number)

    def get_experiment_config(self, experiment_block_number):
        name = self.get_experiment_block_name(experiment_block_number)
        config = self.nix_file.blocks[name].metadata.sections['app_config']
        return self.read_config(config)

    def get_config_mea_matrix_string(self, experiment_block_number=None):
        from ceed.analysis import read_nix_prop
        if experiment_block_number is not None:
            name = self.get_experiment_block_name(experiment_block_number)
            config = self.nix_file.blocks[name].metadata.sections['app_config']
        else:
            config = self.nix_file.sections['app_config']
        settings = read_nix_prop(config.props['app_settings'])

        m = re.match(self._mea_trans_pat, settings)
        if m is None:
            raise ValueError('Cannot find mea transform in the settings')

        return m.group(1)

    def set_config_mea_matrix_string(
            self, experiment_block_number, config_string, new_config_string):
        from ceed.analysis import read_nix_prop
        if config_string == new_config_string:
            raise ValueError('New and old MEA transform are identical')

        name = self.get_experiment_block_name(experiment_block_number)
        config = self.nix_file.blocks[name].metadata.sections['app_config']
        settings = read_nix_prop(config.props['app_settings'])
        settings_new = settings.replace(config_string, new_config_string)
        if settings == settings_new:
            raise ValueError('Could not find the MEA matrix in the config')

        config['app_settings'] = settings_new
        self.has_unsaved = True
        self.dispatch(
            'on_experiment_change', 'experiment_mea_settings',
            experiment_block_number)

    def get_experiment_metadata(self, experiment_block_number):
        from ceed.analysis import CeedDataReader
        name = self.get_experiment_block_name(experiment_block_number)
        block = self.nix_file.blocks[name]

        t = block.metadata['save_time'] if 'save_time' in block.metadata else 0
        notes = block.metadata['notes'] if 'notes' in block.metadata else ''
        if len(block.data_arrays['frame_time']):
            duration_sec = (block.data_arrays['frame_time'][-1] -
                            block.data_arrays['frame_time'][0])
        else:
            duration_sec = 0

        metadata = {
            'stage': block.metadata['stage'],
            'save_time': float(t),
            'image': CeedDataReader.read_image_from_block(block),
            'notes': notes,
            'duration_frames': len(block.data_arrays['frame_time']),
            'duration_sec': float(duration_sec),
            'experiment_number': experiment_block_number,
            'config': self.get_experiment_config(experiment_block_number),
            'mea_config': self.get_config_mea_matrix_string(
                experiment_block_number),
        }
        return metadata

    def get_saved_image(self, image_num):
        from ceed.analysis import CeedDataReader
        block = self.nix_file.blocks['fluorescent_images']
        group = block.groups['fluorescent_image_{}'.format(image_num)]
        data = {
            'save_time': float(group.metadata['save_time']),
            'notes': group.metadata['notes'],
            'image': CeedDataReader.read_image_from_block(
                block, postfix='_{}'.format(image_num)),
            'image_num': image_num,
        }
        return data

    @staticmethod
    def get_blocks_experiment_numbers(blocks, ignore_list=None):
        experiments = []
        ignore_list = set(map(str, ignore_list or []))

        for block in blocks:
            m = re.match(CeedDataWriterBase._experiment_pat, block.name)
            if m is not None and m.group(1) not in ignore_list:
                experiments.append(m.group(1))

        return list(sorted(experiments, key=int))

    @staticmethod
    def get_experiment_block_name(experiment_num):
        return 'experiment{}'.format(experiment_num)

    def ensure_array_size(self, used, allocated, added=1):
        required = used + added
        if allocated >= required:
            return
        allocated = allocated = (required >> 3) + 6 + required

    @app_error
    def collect_experiment(self, block, shapes, frame_bits, frame_counter,
                           frame_time, frame_time_counter, queue, lock,
                           led_state):
        while True:
            try:
                msg, value = queue.get()
            except Empty:
                continue
            eof = msg == 'eof'

            if eof:
                value()
                break
            elif msg == 'frame':
                counter_bits, shape_rgba = value
                lock.acquire()
                try:
                    frame_counter.append(counter_bits['count'])
                    frame_bits.append(counter_bits['bits'])

                    for name, arr in shapes.items():
                        arr.append(shape_rgba[name])
                finally:
                    lock.release()
            elif msg == 'frame_flip':
                lock.acquire()
                try:
                    frame_time_counter.append(value['count'])
                    frame_time.append(value['t'])
                finally:
                    lock.release()
            elif msg == 'led_state':
                count, r, g, b = value
                rec = np.rec.fromarrays((
                    np.array([count], dtype=np.uint64),
                    np.array([r], dtype=np.uint8),
                    np.array([g], dtype=np.uint8),
                    np.array([b], dtype=np.uint8)),
                    names=('frame', 'r', 'g', 'b'))

                lock.acquire()
                try:
                    led_state.append(rec)
                finally:
                    lock.release()
            elif msg == 'debug_data':
                name, data = value
                arr_name = f'debug_{name}'

                lock.acquire()
                try:
                    if arr_name in block.data_arrays:
                        block.data_arrays[arr_name].append(data)
                    else:
                        block.create_data_array(arr_name, 'debug', data=data)
                finally:
                    lock.release()

    def prepare_experiment(self, stage_name, used_shapes):
        self.stop_experiment()

        i = len(self.get_experiment_numbers())
        block = self.nix_file.create_block(
            self.get_experiment_block_name(i), 'experiment_data')

        sec = self.nix_file.create_section(
            'experiment{}_metadata'.format(i), 'metadata')
        sec['stage'] = stage_name

        import ceed
        sec['ceed_version'] = ceed.__version__
        sec['save_time'] = '{}'.format(time.time())
        sec['notes'] = ''

        config = sec.create_section('app_config', 'configuration')
        self.write_config(config)

        block.metadata = sec

        app = App.get_running_app()
        if app.player and app.player.last_image:
            self.write_fluorescent_image(block, app.player.last_image)

        shapes = {}
        for name in used_shapes:
            shapes[name] = block.create_data_array(
                'shape_{}'.format(name), 'shape_data', dtype=np.float16,
                data=np.zeros((0, 4)))

        frame_bits = block.create_data_array(
                'frame_bits', 'port_state', dtype=np.uint32, data=[])
        frame_counter = block.create_data_array(
                'frame_counter', 'counter', dtype=np.uint64, data=[])
        frame_time = block.create_data_array(
                'frame_time', 'time', dtype=np.float64, data=[])
        frame_time_counter = block.create_data_array(
                'frame_time_counter', 'counter', dtype=np.uint64, data=[])

        led_state = block.create_data_array(
            'led_state', 'led_state',
            data=np.rec.fromarrays(
                (np.empty(
                    (0, ), dtype=np.uint64), np.empty((0, ), dtype=np.uint8),
                 np.empty(
                     (0, ), dtype=np.uint8), np.empty((0, ), dtype=np.uint8)),
                names=('frame', 'r', 'g', 'b')))

        self.data_queue = Queue()
        self.data_lock = RLock()
        t = self.data_thread = Thread(
            target=self.collect_experiment, name='data_collection',
            args=(block, shapes, frame_bits, frame_counter, frame_time,
                  frame_time_counter, self.data_queue, self.data_lock,
                  led_state))
        t.start()

    def stop_experiment(self):
        if not self.data_thread:
            return

        block = self.nix_file.blocks[-1]
        name = block.name

        def wait_for_stop():
            def notify(*largs):
                self.dispatch(
                    'on_experiment_change', 'experiment_ended',
                    name[len('experiment'):])
            Clock.schedule_once(notify)

        self.data_queue.put_nowait(('eof', wait_for_stop))
        self.data_queue = self.data_thread = self.data_lock = None
        self.dispatch(
            'on_experiment_change', 'experiment_stop',
            block.name[len('experiment'):])

    def add_frame(self, data):
        if self.data_queue:
            self.data_queue.put_nowait(('frame', data))
            self.has_unsaved = True

    def add_frame_flip(self, data):
        if self.data_queue:
            self.data_queue.put_nowait(('frame_flip', data))
            self.has_unsaved = True

    def add_led_state(self, count, r, g, b):
        if self.data_queue:
            self.data_queue.put_nowait(('led_state', (count, r, g, b)))
            self.has_unsaved = True

    def add_debug_data(self, name, data):
        if self.data_queue:
            self.data_queue.put_nowait(('debug_data', (name, data)))
            self.has_unsaved = True


class DataSerializerBase(EventDispatcher):
    """To facilitate temporal data alignment between the Ceed data (shape
    intensity) and MCS (electrode data), Ceed outputs a bit pattern in the
    corner pixel that is output by the projector controller as a bit pattern
    on its digital port, and is recorded by MCS.

    Post-experiment we locate each Ceed frame in the recorded MCS data by
    locating the Ceed frame bit-pattern in the MCS data, thereby locating the
    sample index in the MCS data, corresponding to the Ceed frame.

    While we have 24 bits to use from the Ceed side, MCS can only record upto
    16 bits, possibly less depending on the config, so we have to be flexible
    on the bit-pattern to send. Our bit-pattern has three components:

    1. A clock bit that alternates low/high for each frame, starting with high.
       The clock bit is set by :attr:`clock_idx`.
    2. Some bits are dedicated to simple short counter that overflows back to
       zero at the top. E.g. if we dedicate 3-bits for the counter, then it'll
       go 0,1,...,6,7,0,1,...,6,7,0,1,... etc.

       The bits used by the short counter is set with
       :attr:`short_count_indices`. The length of the list determines the size
       of the counter, as explained.

       As explained in #3 below, there's a global counter that increments for
       each frame. Potentially, this increment could be larger than one
       (although currently it's always one). The short counter increments the
       counter with this frame increment, so it could potentially go
       0,1,3,...,6,7,0,1,2,... if the third frame incremented the global
       counter by two instead of one.
    3. Some bits are dedicated to a long term counter (and initial
       handshaking). E.g. if the counter is a 32-bit int, then the counter will
       just count up until 2 ^ 32 when it overflows back to zero.

       However, the counter could be 32-bits wide, but the number of bits
       available on the projector is much less. So, we split up the counter
       across multiple frames. E.g. if :attr:`counter_bit_width` is 32 and
       :attr:`count_indices` has only, say, 4 bits available. Then we split
       the 32 bits into ``32 / 4`` transactions. However, each transaction is
       sent twice, the value and its one's complement (aligning with the clock
       being low), except for the first transaction of each number and for the
       initial handshake bytes sent, where it's sent twice, identically,
       without inverting to one's complement. So, the total number of frames
       required is ``32 / 4 * 2 = 16`` frames.

       So, overall for this example, if the first count value starts at zero
       and increments with each frame, then the first 16 frames sends the
       number 0. When it's done, the counter is at 16, so we send the number 16
       over the next 16 frames, then the number 32 over the next 16 frames etc.

       The counter is broken and sent starting from the least significant
       (lower) bits to the most significant (upper) bits.

       At the start, we also optionally send an arbitrary sequence of bytes
       to help identify experimental specific metadata as described in
       :meth:`get_bits`. The counter then starts sending its current value when
       the bytes are done. The length of the bytes is sent before the bytes
       (zero is sent if empty).
    """

    _config_props_ = (
        'counter_bit_width', 'clock_idx', 'count_indices',
        'short_count_indices', 'projector_to_aquisition_map')

    counter_bit_width: int = NumericProperty(32)
    """The number of bits in the long counter, as described in
    :class:`DataSerializerBase`.

    Must be a multiple of 8 (to align with a byte).
    """

    clock_idx: int = NumericProperty(2)
    """The bit index to use for the clock.

    A number between 0-23, inclusive.
    """

    count_indices: List[int] = ListProperty([19, 20])
    """A list of bit indices to use for the long counter as explained in
    :class:`DataSerializerBase`.

    Each item is a number between 0-23, inclusive. Their order is the order of
    the counter bit pattern. The first index is for the first (least
    significant) bit of the counter etc.

    If the length of :attr:`count_indices` doesn't divide
    :attr:`counter_bit_width` exactly, the ends are padded with zeros for those
    bits.
    """

    short_count_indices: List[int] = ListProperty([3, 4, 10, 11, 12, 18])
    """A list of bit indices to use for the short counter as explained in
    :class:`DataSerializerBase`.

    Each item is a number between 0-23, inclusive. Their order is the order of
    the counter bit pattern. E.g. if it was ``[1, 3]``, the the bit pattern for
    just the counter would look like: 0b0000, 0b0010, 0b1000, 0b1010, 0b0000,
    0b0010, 0b1000...
    """

    projector_to_aquisition_map: Dict[int, int] = DictProperty(
        {2: 0, 3: 1, 4: 2, 10: 3, 11: 4, 12: 5, 18: 6, 19: 7, 20: 8})
    """Maps the bit indices used by Ceed to the corresponding bit indices used
    by MCS. It is required to be able to align the two systems.

    I.e. if port zero of the projector is connected to port 3 of the MCS
    controller, then this would be ``{0: 3}``.
    """

    def _validate_config(self):
        if self.counter_bit_width % 8:
            raise ValueError('counter_bit_width must be a multiple of 8')
        if self.counter_bit_width > 64:
            raise ValueError('counter_bit_width can be at most 64')
        if self.counter_bit_width < 8:
            raise ValueError('counter_bit_width must be at least 8')
        if not self.count_indices:
            raise ValueError('The counter bits were not provided')
        if not self.short_count_indices:
            raise ValueError('The short counter bits were not provided')

        clock_idx = self.clock_idx
        if (clock_idx in self.count_indices or
                clock_idx in self.count_indices):
            raise ValueError('The clock bit is re-used in the counter')
        if set(self.count_indices) & set(self.short_count_indices):
            raise ValueError('The counter and short counter bits overlap')

        mapping = self.projector_to_aquisition_map
        for b in [clock_idx] + self.short_count_indices + self.count_indices:
            if b not in mapping:
                raise ValueError(
                    f'bit {b} is used but is not listed in the acquisition map')

    def get_bits(
            self, config_bytes: bytes = b'') -> Generator[int, int, None]:
        """A generator that yields a 24 bit value for each clock frame, that is
        to be used as a RGB value which is then output to the projector
        controller by the hardware and connected to the MCS controller.

        At each iteration, the generator gets send the current frame count
        (typically increments by one) and it yields the RGB value to use for
        that frame.

        If ``n_sub_frames`` is more than 1, the digital IO is simply duplicated
        for those sub-frames by Ceed increasing the number of frames. However,
        :meth:`DataSerializerBase.get_bits` gets only called once per group of
        sub-frames. So while the short counter is only incremented once per
        group of sub-frames, the counter does get incremented once per
        sub-frame.

        :param config_bytes: An optional bytes object to be sent. If provided,
            it will be padded with zeros to :attr:`counter_bit_width` divided
            by eight. E.g. if the length of ``config_bytes`` is 15 and
            :attr:`counter_bit_width` is 32, it'll be padded with one zero byte
            at the end.
        """
        self._validate_config()

        # the clock
        clock_bit_set = 1 << self.clock_idx
        clock_state = 0

        count_bits_set = [1 << v for v in self.count_indices]
        short_bits_set_idx = [
            (i, 1 << v) for i, v in enumerate(self.short_count_indices)]

        short_max_value = 2 ** len(short_bits_set_idx)
        n_count_bits = len(count_bits_set)

        # the counter is broken across multiple frames, each part sent twice.
        # this maps all the count bits parts to the rgb bits (which are the
        # same for each part). Pad the ends to counter_bit_width
        n_samples_per_count = int(ceil(self.counter_bit_width / n_count_bits))
        # output bits and corresponding input indices. Each item is the list
        # for that sample sent. E.g. [[(0b01, 0), (0b10, 1)],
        # [(0b01, 0), (0b10, 1)], [(0b01, 2), (0b10, 3)],
        # [(0b01, 2), (0b10, 3)]], each duplicated for high/low clock
        count_iters = []
        for i in range(n_samples_per_count):
            count_iters.append(list(enumerate(
                count_bits_set, i * n_count_bits)))
            count_iters.append(list(enumerate(
                count_bits_set, i * n_count_bits)))

        # pad config bytes to exact multiple of counter_bit_width // 8
        n_bytes_per_count = self.counter_bit_width // 8
        # pad config bytes to n_bytes_per_count
        pad = n_bytes_per_count - len(config_bytes) % n_bytes_per_count
        config_bytes += b'\0' * pad
        if 2 ** self.counter_bit_width - 1 < len(config_bytes):
            raise ValueError(
                'Cannot transmit config, its too long for counter_bit_width')

        # send the size of the config, followed by the config bytes in groups
        # of counter_bit_width
        config_bytes = [
            len(config_bytes) // n_bytes_per_count] + \
            list(struct.unpack(
                f'<{len(config_bytes) // n_bytes_per_count}L', config_bytes))
        sending_config = True

        value = 0
        count_val = 0
        short_count = 0
        while True:
            first_transaction = True
            for k, data in enumerate(count_iters):
                odd_frame = bool(k % 2)
                # yield last value, wait to get new frame count
                count = yield value

                # reset value to just the clock state (which alternates)
                value = clock_state = clock_state ^ clock_bit_set

                for i, v in short_bits_set_idx:
                    if (1 << i) & short_count:
                        value |= v
                short_count = (short_count + 1) % short_max_value

                # update count number sent only at the first transaction
                if first_transaction:
                    # either send the current count, or the next config value
                    # until transactions done
                    if config_bytes:
                        count_val = config_bytes.pop(0)
                        sending_config = True
                    else:
                        count_val = count
                        sending_config = False
                    first_transaction = False

                for i, v in data:
                    if not odd_frame or k == 1 or sending_config:
                        # sending number itself
                        # if bit i of the n-bit number is 1, set the rgb bit
                        # corresponding to the part containing i
                        if (1 << i) & count_val:
                            value |= v
                    else:
                        # sending one's complement
                        if not ((1 << i) & count_val):
                            value |= v

    def num_ticks_handshake(self, config_len, n_sub_frames):
        """Gets the number of frames required to transmit the handshake
        signature (i.e. config bytes) of the experiment as provided to
        :meth:`get_bits` with the given ``config_bytes``.

        See also :func:`num_ticks_handshake`.

        :param config_len: The number of config **bytes** being sent (not
            including padding bytes).
        :param n_sub_frames: the number of sub-frames in each frame.
        """
        return num_ticks_handshake(
            self.counter_bit_width, self.count_indices, config_len,
            n_sub_frames)


def num_ticks_handshake(
        counter_bit_width, count_indices, config_len, n_sub_frames):
    """Gets the number of frames required to transmit the handshake
    signature (i.e. config bytes) of the experiment as provided to
    :meth:`DataSerializerBase.get_bits` with the given ``config_bytes``.

    :param counter_bit_width: See :attr:`DataSerializerBase.counter_bit_width`.
    :param count_indices: See :attr:`DataSerializerBase.count_indices`.
    :param config_len: The number of config **bytes** being sent (not including
        padding bytes).
    :param n_sub_frames: the number of sub-frames in each frame. E.g. in quad4x
        mode, each frame is actually 4 frames, but the digital IO is the same
        for all of them.
    """
    n_bytes_per_value = counter_bit_width // 8
    # the message + padding + message length
    n_values = int(ceil(config_len / n_bytes_per_value)) + 1

    frames_per_int = int(ceil(counter_bit_width / len(count_indices)))
    # each frame is duplicated and sub-frames have the same value
    return n_values * frames_per_int * 2 * n_sub_frames


def num_ticks_handshake_1_0_0_dev0(
        counter_bit_width, count_indices, config_len, n_sub_frames):
    """Same as :func:`num_ticks_handshake`, but it returns the value used for
    ceed version 1.0.0.dev0.
    """
    n_bytes_per_count = counter_bit_width // 8
    # the message + padding in bytes
    config_len += n_bytes_per_count - (config_len % n_bytes_per_count)
    config_len //= n_bytes_per_count  # in
    config_len += 1  # the message length byte

    ticks_per_int = int(ceil(counter_bit_width / len(count_indices)))
    return config_len * ticks_per_int * 2 + 2
