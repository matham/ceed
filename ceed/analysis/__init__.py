import numpy as np
from fractions import Fraction
import nixio as nix
import re
from cplcom.utils import yaml_dumps, yaml_loads
from cplcom.config import apply_config
from cplcom.player import Player
from ffpyplayer.pic import Image
from ffpyplayer.writer import MediaWriter

from ceed.function import FunctionFactoryBase, register_all_functions
from ceed.stage import StageFactoryBase, StageDoneException
from ceed.shape import CeedPaintCanvasBehavior
from ceed.storage.controller import DataSerializerBase
from ceed.view.controller import ViewControllerBase


class CeedDataReader(object):

    filename = ''

    shapes_intensity = {}

    electrodes_data = {}

    electrodes_metadata = {}

    electrode_dig_data = None

    electrode_intensity_alignment = None

    led_state = None

    view_controller = None

    data_serializer = None

    function_factory = None

    stage_factory = None

    shape_factory = None

    experiment_stage_name = ''

    ceed_version = ''

    experiment = 0

    _nix_file = None

    _block = None

    _experiment_pat = re.compile('^experiment([0-9]+)$')

    def __init__(self, filename, **kwargs):
        super(CeedDataReader, self).__init__(**kwargs)
        self.filename = filename

    def open_h5(self):
        if self._nix_file is not None:
            raise Exception('File already open')

        self._nix_file = nix.File.open(
            self.filename, nix.FileMode.ReadOnly)

    def get_experiments(self):
        experiments = []
        for block in self._nix_file.blocks:
            m = re.match(self._experiment_pat, block.name)
            if m is not None:
                experiments.append(m.group(1))
        return list(sorted(experiments, key=int))

    def read_experiment(self, experiment):
        self._block = block = self._nix_file.blocks[
            'experiment{}'.format(experiment)]
        section = self._nix_file.sections[
            'experiment{}_metadata'.format(experiment)]
        self.experiment = experiment

        self.experiment_stage_name = section['stage']
        self.ceed_version = section['ceed_version']
        config = section.sections['app_config']

        config_data = {}
        for prop in config.props:
            config_data[prop.name] = yaml_loads(prop.values[0].value)

        view = self.view_controller = ViewControllerBase()
        ser = self.data_serializer = DataSerializerBase()
        func = self.function_factory = FunctionFactoryBase()
        register_all_functions(func)
        shape = self.shape_factory = CeedPaintCanvasBehavior(knsname='painter')
        stage = self.stage_factory = StageFactoryBase(
            function_factory=func, shape_factory=shape)

        apply_config(config_data['app_settings'], {
            'view': view, 'serializer': ser, 'function': func})
        self.populate_config(config_data, shape, func, stage)

        data = self.shapes_intensity = {}
        for item in block.data_arrays:
            if not item.name.startswith('shape_'):
                continue
            data[item.name[6:]] = item

        self.led_state = block.data_arrays['led_state']

        self.electrode_intensity_alignment = self._nix_file.blocks[
            'ceed_mcs_alignment'].data_arrays[
            'experiment_{}'.format(experiment)]

        mcs_block = self._nix_file.blocks['mcs_data']
        mcs_metadata = mcs_block.metadata
        self.electrode_dig_data = None
        if 'digital_io' in mcs_block.data_arrays:
            self.electrode_dig_data = mcs_block.data_arrays['digital_io']

        electrode_data = self.electrodes_data = {}
        electrodes_metadata = self.electrodes_metadata = {}
        for item in mcs_block.data_arrays:
            if not item.name.startswith('electrode_'):
                continue
            electrode_data[item.name[10:]] = item

            electrodes_metadata[item.name[10:]] = electrode_metadata = {}
            for prop in mcs_metadata.sections[item.name].props:
                electrode_metadata[prop.name] = yaml_loads(prop.values[0].value)

    def get_fluorescent_image(self):
        return self.read_fluorescent_image_from_block(self._block)

    def save_flourescent_image(self, filename, img=None, codec='bmp'):
        if img is None:
            img = self.get_fluorescent_image()
        Player.save_image(
            filename, img, codec=codec, pix_fmt=img.get_pixel_format())

    def get_electrode_offset_scale(self, electrode):
        metadata = self.electrodes_metadata[electrode]
        return (
            float(metadata['ADZero']),
            float(metadata['ConversionFactor']) *
            10. ** float(metadata['Exponent']))

    def generate_movie(
            self, filename, out_fmt='yuv420p', codec='libx264',
            lib_opts={'crf': '0'}, start=None, end=None):
        from kivy.graphics import (
            Canvas, Translate, Fbo, ClearColor, ClearBuffers, Scale)
        from kivy.core.window import Window

        count = 0
        rate = float(self.view_controller.frame_rate)
        rate_int = int(rate)
        w, h = Window.size = (
            self.view_controller.screen_width,
            self.view_controller.screen_height)

        stream = {
            'pix_fmt_in': 'rgba', 'pix_fmt_out': out_fmt,
            'width_in': w, 'height_in': h, 'width_out': w,
            'height_out': h, 'codec': codec, 'frame_rate': (rate_int, 1)}
        writer = MediaWriter(filename, [stream], fmt='mp4', lib_opts=lib_opts)

        fbo = Fbo(size=(w, h), with_stencilbuffer=True)

        with fbo:
            ClearColor(0, 0, 0, 1)
            ClearBuffers()
            Scale(1, -1, 1)
            Translate(0, -h, 0)

        fbo.draw()
        img = Image(plane_buffers=[fbo.pixels], pix_fmt='rgba', size=(w, h))
        writer.write_frame(img, count / rate)

        tick = self.stage_factory.tick_stage(self.experiment_stage_name)
        shape_views = self.stage_factory.get_shapes_gl_color_instructions(
            fbo, 'stage_replay')

        while True:
            count += 1
            try:
                next(tick)
                t = Fraction(count, rate_int)
                if end is not None and end < t:
                    break
                shape_values = tick.send(t)
            except StageDoneException:
                break

            if start is not None and start < t:
                continue
            self.stage_factory.fill_shape_gl_color_values(
                shape_views, shape_values)

            fbo.draw()
            img = Image(plane_buffers=[fbo.pixels], pix_fmt='rgba', size=(w, h))
            writer.write_frame(img, count / rate)

    @staticmethod
    def populate_config(
            settings, shape_factory, function_factory, stage_factory):
        old_to_new_name_shape_map = {}
        shape_factory.set_state(settings['shape'], old_to_new_name_shape_map)

        id_to_func_map = {}
        old_id_map = {}
        old_to_new_name = {}

        f1 = function_factory.recover_funcs(
            settings['function'], id_to_func_map, old_id_map, old_to_new_name)
        f2 = stage_factory.recover_stages(
            settings['stage'], id_to_func_map, old_id_map=old_id_map,
            old_to_new_name_shape_map=old_to_new_name_shape_map)

        old_to_new_id_map = {v: k for k, v in old_id_map.items()}

        id_map = settings['func_id_map']
        id_map_new = {old_to_new_id_map[a]: old_to_new_id_map[b]
                      for a, b in id_map.items()
                      if a in old_to_new_id_map and b in old_to_new_id_map}

        for f in f1[0] + f2[0]:
            for func in f.get_funcs():
                func.finalize_func_state(
                    id_map_new, id_to_func_map, old_to_new_name)

    @staticmethod
    def read_fluorescent_image_from_block(block):
        try:
            group = block.groups['fluorescent_image']
        except KeyError:
            return None

        planes = [np.array(d).tobytes() for d in group.data_arrays]
        img = Image(plane_buffers=planes, pix_fmt=group.metadata['pix_fmt'],
                    size=yaml_loads(group.metadata['size']))
        return img


if __name__ == '__main__':
    f = CeedDataReader(r'E:\test.h5')
    f.open_h5()
    f.read_experiment(0)
    f.generate_movie(r'E:\test.mp4')
