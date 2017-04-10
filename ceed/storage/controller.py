import json
import nixio as nix
from os.path import exists, basename, splitext, split, join, isdir
from os import remove
from tempfile import NamedTemporaryFile
from shutil import copy2
from math import ceil
from threading import Thread, RLock
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
import numpy as np
from ffpyplayer.pic import Image

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, ListProperty, \
    DictProperty, BooleanProperty
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.behaviors.knspace import knspace
from kivy.logger import Logger

from cplcom.config import byteify
from cplcom.utils import json_dumps, json_loads

from ceed.function import FunctionFactory, FuncBase
from ceed.stage import StageFactory
from ceed.shape import get_painter


class CeedDataBase(EventDispatcher):

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

    def __init__(self, **kwargs):
        super(CeedDataBase, self).__init__(**kwargs)
        Clock.schedule_interval(self.backup_file, self.backup_interval)

    @staticmethod
    def gather_config_data_dict():
        from ceed.view.controller import ViewController
        data = {}
        data['shape'] = get_painter().save_state()
        func_id_map = {}
        data['function'] = FunctionFactory.save_funcs(func_id_map)[0]
        data['stage'] = StageFactory.save_stages(func_id_map)[0]
        data['func_id_map'] = func_id_map
        data['view_controller'] = ViewController.save_state()

        return data

    @staticmethod
    def clear_existing_config_data():
        StageFactory.clear_stages()
        get_painter().remove_all_groups()
        get_painter().delete_all_shapes()
        FunctionFactory.clear_funcs()

    @staticmethod
    def apply_config_data_dict(data):
        from ceed.view.controller import ViewController
        ViewController.set_state(data['view_controller'])
        get_painter().set_state(data['shape'])

        id_to_func_map = {}
        old_id_map = {}
        f1 = FunctionFactory.recover_funcs(
            data['function'], id_to_func_map, old_id_map=old_id_map)
        f2 = StageFactory.recover_stages(
            data['stage'], id_to_func_map, old_id_map=old_id_map)
        id_map = data['func_id_map']
        id_to_func_map = {
            old_id_map[new_id]: f for new_id, f in id_to_func_map.items()
            if new_id in old_id_map}

        for f in f1[0] + f2[0]:
            FuncBase.set_source_from_id(f, id_map, id_to_func_map)

    def get_filebrowser_callback(
            self, func, check_exists=False, check_unsaved=False):

        def callback(path, selection, filename):
            if not isdir(path) or not filename:
                return
            self.root_path = path
            fname = join(path, filename)

            def discard_callback(discard):
                if check_unsaved and (self.has_unsaved or self.config_changed):
                    if not discard:
                        return
                    self.discard_file()

                if check_exists and exists(fname):
                    def yesno_callback(overwrite):
                        if not overwrite:
                            return
                        func(fname, overwrite)

                    yesno = App.get_running_app().yesno_prompt
                    yesno.msg = ('"{}" already exists, would you like to '
                                 'overwrite it?'.format(fname))
                    yesno.callback = yesno_callback
                    yesno.open()
                else:
                    func(fname)

            if check_unsaved and (self.has_unsaved or self.config_changed):
                yesno = App.get_running_app().yesno_prompt
                yesno.msg = 'There are unsaved changes.\nDiscard them?'
                yesno.callback = discard_callback
                yesno.open()
            else:
                discard_callback(True)
        return callback

    def ui_close(self, app_close=False):
        if self.has_unsaved or self.config_changed:
            def close_callback(discard):
                if discard:
                    self.discard_file()
                    self.close_file()
                    if app_close:
                        App.get_running_app().stop()
                    else:
                        self.create_file('')

            yesno = App.get_running_app().yesno_prompt
            yesno.msg = 'There are unsaved changes.\nDiscard them?'
            yesno.callback = close_callback
            yesno.open()
            return False
        else:
            self.close_file()
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

        f = self.nix_file = nix.File.open(
            self.backup_filename, nix.FileMode.ReadWrite)
        self.config_changed = self.has_unsaved = True
        Logger.debug(
            'Ceed Controller (storage): Created tempfile {}, from existing '
            'file "{}"'.format(self.backup_filename, self.filename))
        self.save()

    def close_file(self):
        if self.data_thread:
            raise TypeError("Cannot close data during an experiment")
        if self.nix_file:
            self.nix_file.close()
            self.nix_file = None

        if (not self.has_unsaved and not self.config_changed or
                not self.filename) and self.backup_filename:
            remove(self.backup_filename)

        Logger.debug(
            'Ceed Controller (storage): Closed tempfile {}, with '
            '"{}"'.format(self.backup_filename, self.filename))
        self.filename = self.backup_filename = ''
        self.config_changed = self.has_unsaved = False

    def import_file(self, filename):
        f = nix.File.open(filename, nix.FileMode.ReadOnly)
        Logger.debug(
            'Ceed Controller (storage): Imported "{}"'.format(self.filename))
        data = {}

        try:
            for prop in f.sections['app_config']:
                data[prop.name] = json_loads(prop.values[0].value)
        except:
            f.close()
            raise
        else:
            f.close()

        self.apply_config_data_dict(data)
        self.config_changed = True

    def discard_file(self):
        if not self.has_unsaved and not self.config_changed:
            return

        if not self.filename:
            self.close_file()
            self.clear_existing_config_data()
            self.create_file('')
            self.has_unsaved = self.config_changed = False
        else:
            f = self.filename
            self.close_file()
            self.open_file(f)

    def save_as(self, filename, overwrite=False):
        if exists(filename) and not overwrite:
            raise ValueError('{} already exists'.format(filename))
        self.save(filename, True)
        self.open_file(filename)

    def save(self, filename=None, force=False):
        if not force and not self.has_unsaved and not self.config_changed:
            return

        try:
            if self.data_lock:
                self.data_lock.acquire()

            self.backup_file()
            filename = filename or self.filename
            if filename:
                copy2(self.backup_filename, filename)
        except:
            if self.data_lock:
                self.data_lock.release()
            raise
        else:
            if self.data_lock:
                self.data_lock.release()

        if filename:
            self.config_changed = self.has_unsaved = False

    def backup_file(self, *largs):
        if not self.nix_file:
            return

        try:
            if self.data_lock:
                self.data_lock.acquire()

            if self.config_changed:
                self.write_config()
            self.nix_file._h5file.flush()
        except:
            if self.data_lock:
                self.data_lock.release()
            raise
        else:
            if self.data_lock:
                self.data_lock.release()

    def write_json_config(self, filename, overwrite=False):
        if exists(filename) and not overwrite:
            raise ValueError('{} already exists'.format(filename))

        data = json_dumps(self.gather_config_data_dict())
        with open(filename, 'w') as fh:
            fh.write(data)

    def read_json_config(self, filename):
        with open(filename, 'r') as fh:
            data = fh.read()
        data = json_loads(data)
        self.apply_config_data_dict(data)
        self.config_changed = True

    def write_config(self, config_section=None):
        config = config_section or self.nix_file.sections['app_config']
        data = self.gather_config_data_dict()
        for k, v in data.items():
            config[k] = json_dumps(v)

    def read_config(self, config_section=None):
        config = config_section or self.nix_file.sections['app_config']
        data = {}
        for prop in config.props:
            data[prop.name] = json_loads(prop.values[0].value)
        return data

    def write_fluorescent_image(self, block, img):
        group = block.create_group('fluorescent_image', 'image')

        config = block.metadata.create_section(
            'fluorescent_image_config', 'image')
        config['size'] = json_dumps(img.get_size())
        config['pix_fmt'] = img.get_pixel_format()
        config['linesizes'] = json_dumps(img.get_linesizes())
        config['buffer_size'] = json_dumps(img.get_buffer_size())
        group.metadata = config

        for i, plane in enumerate(img.to_bytearray()):
            image_data = block.create_data_array(
                    'fluorescent_image_plane_{}'.format(i), 'image',
                    dtype=np.uint32, data=plane)
            group.data_arrays.append(image_data)

    def read_fluorescent_image(self, block):
        try:
            group = block.fluorescent_image
        except AttributeError:
            return None

        planes = [bytes(d) for d in group.data_arrays]
        img = Image(plane_buffers=planes, pix_fmt=group.metadata['pix_fmt'],
                    size=json_loads(group.metadata['size']))
        return img

    def ensure_array_size(self, used, allocated, added=1):
        required = used + added
        if allocated >= required:
            return
        allocated = allocated = (required >> 3) + 6 + required

    def collect_experiment(self, block, shapes, frame_bits, frame_counter,
                           frame_time, frame_time_counter):
        queue = self.data_queue
        lock = self.data_lock
        while True:
            try:
                msg, value = queue.get()
                if msg == 'eof':
                    break
                elif msg == 'frame':
                    count, bits, values = value
                    lock.acquire()
                    frame_counter.append(count)
                    frame_bits.append(bits)
                    for name, r, g, b, a in values:
                        shapes[name].append([[r, g, b, a]])
                    lock.release()
                elif msg == 'frame_flip':
                    count, t = value
                    lock.acquire()
                    frame_time.append(t)
                    frame_time_counter.append(count)
                    lock.release()
            except Empty:
                continue

    def prepare_experiment(self, stage_name):
        self.stop_experiment()

        i = len(self.nix_file.blocks)
        block = self.nix_file.create_block(
            'experiment{}'.format(i), 'experiment_data')

        sec = self.nix_file.create_section(
            'experiment{}_metadata'.format(i), 'metadata')
        sec['stage'] = stage_name
        config = sec.create_section('app_config', 'configuration')
        self.write_config(config)

        block.metadata = sec

        if hasattr(knspace, 'player') and knspace.player.last_image:
            self.write_fluorescent_image(block, knspace.player.last_image)

        shapes = {}
        for shape in get_painter().shapes:
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

        self.data_queue = Queue()
        self.data_lock = RLock()
        t = self.data_thread = Thread(
            target=self.collect_experiment, name='data_collection',
            args=(block, shapes, frame_bits, frame_counter, frame_time,
                  frame_time_counter))
        t.start()

    def stop_experiment(self):
        if not self.data_thread:
            return

        self.data_queue.put_nowait(('eof', None))
        self.data_thread.join()
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


class DataSerializer(EventDispatcher):

    __settings_attrs__ = ('counter_bit_width', 'clock_idx', 'count_indices',
                          'short_count_indices', 'projector_to_aquisition_map')

    counter_bit_width = NumericProperty(16)

    clock_idx = NumericProperty(0)

    count_indices = ListProperty([4, 5])

    short_count_indices = ListProperty([1, 2, 3])

    projector_to_aquisition_map = DictProperty(
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})

    def get_bits(self, last_count):
        clock_base = 1 << self.clock_idx
        clock = 0

        count_i = [1 << v for v in self.count_indices]
        short_i = [(i, 1 << v) for i, v in enumerate(self.short_count_indices)]

        short_n = 2 ** len(short_i)
        short_val = 0

        count_cycles = int(ceil(self.counter_bit_width / float(len(count_i))))
        count_iters = [list(enumerate(count_i))]
        for i in range(count_cycles):
            count_iters.append(list(enumerate(count_i, i * len(count_i))))

        while True:
            first = True
            for data in count_iters:
                count = yield
                value = clock = clock ^ clock_base

                if first:
                    count_val = count
                    first = False

                short_val = (short_val + count - last_count) % short_n
                last_count = count
                for i, v in short_i:
                    if (1 << i) & short_val:
                        value |= v

                for i, v in data:
                    if (1 << i) & count_val:
                        value |= v

                yield value



CeedData = CeedDataBase()
