'''Storage Controller
=======================

Handles all data aspects, from the storage, loading and saving of configuration
data to the acquisition and creation of experimental data.

'''
import nixio as nix
from os.path import exists, basename, splitext, split, join, isdir
from os import remove
import os
from tempfile import NamedTemporaryFile
from shutil import copy2
from math import ceil
from threading import Thread, RLock
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty
import numpy as np
import struct
from ffpyplayer.pic import Image

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, ListProperty, \
    DictProperty, BooleanProperty, ObjectProperty
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.behaviors.knspace import knspace
from kivy.logger import Logger

from cplcom.app import app_error
from cplcom.utils import yaml_dumps, yaml_loads

__all__ = ('CeedDataWriterBase', 'DataSerializerBase')


class CeedDataWriterBase(EventDispatcher):

    __settings_attrs__ = ('root_path', 'backup_interval')

    root_path = StringProperty('')

    backup_interval = NumericProperty(5.)

    filename = StringProperty('')

    backup_filename = ''

    nix_file = None

    has_unsaved = BooleanProperty(False)

    config_changed = BooleanProperty(False)

    data_queue = None

    data_thread = None

    data_lock = None

    func_display_callback = None

    stage_display_callback = None

    shape_display_callback = None

    clear_all_callback = None

    def __init__(self, **kwargs):
        super(CeedDataWriterBase, self).__init__(**kwargs)
        if (not os.environ.get('KIVY_DOC_INCLUDE', None) and
                self.backup_interval):
            Clock.schedule_interval(self.write_changes, self.backup_interval)

    def gather_config_data_dict(self):
        app = App.get_running_app()
        data = {}
        data['shape'] = app.shape_factory.get_state()
        data['function'] = app.function_factory.save_functions()
        data['stage'] = app.stage_factory.save_stages()

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

    def apply_config_data_dict(self, data):
        app = App.get_running_app()
        app_settings = data['app_settings']
        # filter classes that are not of this app
        classes = app.get_config_classes()
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

    def get_filebrowser_callback(
            self, func, check_exists=False, clear_data=False):

        def callback(path, selection, filename):
            if not isdir(path) or not filename:
                return
            self.root_path = path
            fname = join(path, filename)

            def discard_callback(discard):
                if clear_data and not discard:
                    return

                if check_exists and exists(fname):
                    def yesno_callback(overwrite):
                        if not overwrite:
                            return
                        if clear_data:
                            self.close_file(force_remove_autosave=True)
                            self.clear_existing_config_data()
                        func(fname, overwrite)

                    yesno = App.get_running_app().yesno_prompt
                    yesno.msg = ('"{}" already exists, would you like to '
                                 'overwrite it?'.format(fname))
                    yesno.callback = yesno_callback
                    yesno.open()
                else:
                    if clear_data:
                        self.close_file(force_remove_autosave=True)
                        self.clear_existing_config_data()
                    func(fname)

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

    def create_open_file(self, filename):
        if exists(filename):
            self.open_file(filename)
        else:
            self.create_file(filename)

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
            self.backup_filename, nix.FileMode.Overwrite)
        self.config_changed = self.has_unsaved = True
        Logger.debug(
            'Ceed Controller (storage): Created tempfile {}, with file "{}"'.
            format(self.backup_filename, self.filename))

        f.create_section('app_config', 'configuration')
        self.save()

    def open_file(self, filename):
        '''Loads the file's config and opens the file for usage. '''
        self.close_file()

        self.filename = filename

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
            self.backup_filename, nix.FileMode.ReadWrite)
        self.config_changed = self.has_unsaved = True
        Logger.debug(
            'Ceed Controller (storage): Created tempfile {}, from existing '
            'file "{}"'.format(self.backup_filename, self.filename))
        self.save()

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

    def import_file(self, filename):
        '''Loads the file's config data. '''
        f = nix.File.open(filename, nix.FileMode.ReadOnly)
        Logger.debug(
            'Ceed Controller (storage): Imported "{}"'.format(self.filename))
        data = {}

        try:
            for prop in f.sections['app_config']:
                data[prop.name] = yaml_loads(prop.values[0].value)
        except:
            f.close()
            raise
        else:
            f.close()

        self.apply_config_data_dict(data)
        self.config_changed = True

    def discard_file(self,):
        if not self.has_unsaved and not self.config_changed:
            return

        f = self.filename
        self.close_file(force_remove_autosave=True)
        self.clear_existing_config_data()
        if f:
            self.open_file(f)
        else:
            self.create_file('')

    def save_as(self, filename, overwrite=False):
        if exists(filename) and not overwrite:
            raise ValueError('{} already exists'.format(filename))
        self.save(filename, True)
        self.open_file(filename)

    def save(self, filename=None, force=False):
        '''Saves the changes to the autosave and also saves the changes to
        the file in filename (if None saves to the current filename).
        '''
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

    def write_changes(self, *largs):
        '''Writes unsaved changes to the current (autosave) file. '''
        if not self.nix_file:
            return

        try:
            if self.data_lock:
                self.data_lock.acquire()

            if self.config_changed:
                self.write_config()
            self.nix_file._h5file.flush()
        finally:
            if self.data_lock:
                self.data_lock.release()

    def write_yaml_config(self, filename, overwrite=False):
        if exists(filename) and not overwrite:
            raise ValueError('{} already exists'.format(filename))

        data = yaml_dumps(self.gather_config_data_dict())
        with open(filename, 'w') as fh:
            fh.write(data)

    def read_yaml_config(self, filename):
        with open(filename, 'r') as fh:
            data = fh.read()
        data = yaml_loads(data)
        self.apply_config_data_dict(data)
        self.config_changed = True

    def write_config(self, config_section=None):
        config = config_section if config_section is not None else \
            self.nix_file.sections['app_config']
        data = self.gather_config_data_dict()
        for k, v in data.items():
            config[k] = yaml_dumps(v)

    def read_config(self, config_section=None):
        config = config_section if config_section is not None else \
            self.nix_file.sections['app_config']
        data = {}
        for prop in config.props:
            data[prop.name] = yaml_loads(prop.values[0].value)
        return data

    def load_last_fluorescent_image(self, filename):
        from ceed.analysis import CeedDataReader
        f = nix.File.open(filename, nix.FileMode.ReadOnly)
        Logger.debug(
            'Ceed Controller (storage): Importing fluorescent image from '
            '"{}"'.format(self.filename))

        try:
            if not f.blocks:
                raise ValueError('Image not found in {}'.format(filename))

            for block in reversed(f.blocks):
                img = CeedDataReader.read_fluorescent_image_from_block(block)
                if img is not None:
                    return img

            raise ValueError('Image not found in {}'.format(filename))
        finally:
            f.close()

    def write_fluorescent_image(self, block, img):
        group = block.create_group('fluorescent_image', 'image')

        config = block.metadata.create_section(
            'fluorescent_image_config', 'image')
        config['size'] = yaml_dumps(img.get_size())
        config['pix_fmt'] = img.get_pixel_format()
        config['linesizes'] = yaml_dumps(img.get_linesizes())
        config['buffer_size'] = yaml_dumps(img.get_buffer_size())
        group.metadata = config

        for i, plane in enumerate(img.to_bytearray()):
            image_data = block.create_data_array(
                    'fluorescent_image_plane_{}'.format(i), 'image',
                    dtype=np.uint8, data=plane)
            group.data_arrays.append(image_data)

    def ensure_array_size(self, used, allocated, added=1):
        required = used + added
        if allocated >= required:
            return
        allocated = allocated = (required >> 3) + 6 + required

    @app_error
    def collect_experiment(self, block, shapes, frame_bits, frame_counter,
                           frame_time, frame_time_counter, queue, lock,
                           led_state):

        frame_count_buf = np.zeros((512, ), dtype=np.uint64)
        bits_buf = np.zeros((512, ), dtype=np.uint32)
        frame_i = 0  # counter

        frame_vals_buf = {
            name: np.zeros((512, 4), dtype=np.float16) for name in shapes}
        frame_vals_i = {name: 0 for name in shapes}  # counter

        flip_count_buf = np.zeros((512, ), dtype=np.uint64)
        flip_t_buf = np.zeros((512, ), dtype=np.float64)
        flip_i = 0  # counter

        while True:
            try:
                msg, value = queue.get()
            except Empty:
                continue
            eof = msg == 'eof'

            assert frame_i <= 512
            if frame_i == 512 or eof:
                lock.acquire()
                try:
                    frame_counter.append(frame_count_buf[:frame_i])
                    frame_bits.append(bits_buf[:frame_i])
                    frame_count_buf[:] = bits_buf[:] = 0

                    for name, vals in frame_vals_buf.items():
                        shapes[name].append(vals[:frame_vals_i[name], :])
                        frame_vals_i[name] = vals[:] = 0
                finally:
                    lock.release()

                frame_i = 0

            assert flip_i <= 512
            if flip_i == 512 or eof:
                lock.acquire()
                try:
                    frame_time_counter.append(flip_count_buf[:flip_i])
                    frame_time.append(flip_t_buf[:flip_i])
                    flip_count_buf[:] = flip_t_buf[:] = 0
                finally:
                    lock.release()

                flip_i = 0

            if eof:
                break
            elif msg == 'frame':
                count, bits, values = value
                frame_count_buf[frame_i] = count
                bits_buf[frame_i] = bits
                frame_i += 1

                for name, r, g, b, a in values:
                    frame_vals_buf[name][frame_vals_i[name], :] = r, g, b, a
                    frame_vals_i[name] += 1
            elif msg == 'frame_flip':
                count, t = value
                flip_count_buf[flip_i] = count
                flip_t_buf[flip_i] = t
                flip_i += 1
            elif msg == 'led_state':
                count, r, g, b = value
                rec = np.rec.fromarrays((
                    np.array([count], dtype=np.uint64),
                    np.array([r], dtype=np.uint8),
                    np.array([g], dtype=np.uint8),
                    np.array([b], dtype=np.uint8)),
                    names=('frame', 'r', 'g', 'b'))
                led_state.append(rec)

    def prepare_experiment(self, stage_name):
        self.stop_experiment()

        i = len(self.nix_file.blocks)
        block = self.nix_file.create_block(
            'experiment{}'.format(i), 'experiment_data')

        sec = self.nix_file.create_section(
            'experiment{}_metadata'.format(i), 'metadata')
        sec['stage'] = stage_name

        import ceed
        sec['ceed_version'] = ceed.__version__

        config = sec.create_section('app_config', 'configuration')
        self.write_config(config)

        block.metadata = sec

        if hasattr(knspace, 'player') and knspace.player.last_image:
            self.write_fluorescent_image(block, knspace.player.last_image)

        shapes = {}
        for shape in App.get_running_app().shape_factory.shapes:
            shapes[shape.name] = block.create_data_array(
                'shape_{}'.format(shape.name), 'shape_data', dtype=np.float16,
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

        self.data_queue.put_nowait(('eof', None))
        self.data_queue = self.data_thread = self.data_lock = None

    def add_frame(self, count, bits, values):
        if not count:
            return
        if self.data_queue:
            self.data_queue.put_nowait(('frame', (count, bits, values)))
            self.has_unsaved = True

    def add_frame_flip(self, count, t):
        if not count:
            return
        if self.data_queue:
            self.data_queue.put_nowait(('frame_flip', (count, t)))
            self.has_unsaved = True

    def add_led_state(self, count, r, g, b):
        if self.data_queue:
            self.data_queue.put_nowait(('led_state', (count, r, g, b)))
            self.has_unsaved = True


class DataSerializerBase(EventDispatcher):

    __settings_attrs__ = ('counter_bit_width', 'clock_idx', 'count_indices',
                          'short_count_indices', 'projector_to_aquisition_map')

    counter_bit_width = NumericProperty(32)

    clock_idx = NumericProperty(2)

    count_indices = ListProperty([19, 20])

    short_count_indices = ListProperty([3, 4, 10, 11, 12, 18])

    projector_to_aquisition_map = DictProperty(
        {2: 0, 3: 1, 4: 2, 10: 3, 11: 4, 12: 5, 18: 6, 19: 7, 20: 8})

    def get_bits(self, last_count, config_bytes=b''):
        clock_base = 1 << self.clock_idx
        clock = 0

        count_i = [1 << v for v in self.count_indices]
        short_i = [(i, 1 << v) for i, v in enumerate(self.short_count_indices)]

        short_n = 2 ** len(short_i)
        short_val = 0

        count_cycles = int(ceil(self.counter_bit_width / float(len(count_i))))
        count_iters = []
        for i in range(count_cycles):
            count_iters.append(list(enumerate(count_i, i * len(count_i))))
            count_iters.append(list(enumerate(count_i, i * len(count_i))))

        config_bytes = [
            len(config_bytes)] + \
            list(struct.unpack('<{}L'.format(len(config_bytes) // 4),
                               config_bytes))
        sending_config = bool(config_bytes)

        while True:
            first = True
            for k, data in enumerate(count_iters):
                count = yield
                value = clock = clock ^ clock_base

                short_val = (short_val + count - last_count) % short_n
                last_count = count
                for i, v in short_i:
                    if (1 << i) & short_val:
                        value |= v

                if first:
                    if config_bytes:
                        count_val = config_bytes.pop(0)
                        sending_config = True
                    else:
                        count_val = count
                        sending_config = False
                    first = False

                for i, v in data:
                    if ((not k % 2 or k == 1 or sending_config) and
                        ((1 << i) & count_val)) or \
                            k % 2 and not ((1 << i) & count_val):
                        value |= v

                yield value
            else:
                pass
