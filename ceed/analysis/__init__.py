import math
import sys
import scipy.io
import numpy as np
from fractions import Fraction
import sys
import logging
import nixio as nix
import re
from numpy.lib.format import open_memmap

from cplcom.utils import yaml_dumps, yaml_loads
from cplcom.config import apply_config
from cplcom.player import Player
from ffpyplayer.pic import Image, SWScale
from ffpyplayer.tools import get_best_pix_fmt
from tqdm import tqdm
from ffpyplayer.writer import MediaWriter

from ceed.function import FunctionFactoryBase, register_all_functions
from ceed.stage import StageFactoryBase, StageDoneException
from ceed.shape import CeedPaintCanvasBehavior
from ceed.storage.controller import DataSerializerBase
from ceed.view.controller import ViewControllerBase


class CallableGen(object):

    def __init__(self, gen):
        self.gen = gen
        next(gen)

    def __call__(self, i):
        self.gen.send(i)


def callable_gen(gen):
    def outer_func(*l, **kw):
        def inner_func(*largs, **kwargs):
            return CallableGen(gen(*l, *largs, **kwargs, **kw))
        return inner_func
    return outer_func


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

    _YUV_RGB_FS = '''
    $HEADER$
    uniform sampler2D tex_y;
    uniform sampler2D tex_u;
    uniform sampler2D tex_v;

    void main(void) {
        float y = texture2D(tex_y, tex_coord0).r;
        float u = texture2D(tex_u, tex_coord0).r - 0.5;
        float v = texture2D(tex_v, tex_coord0).r - 0.5;
        float r = y + 1.402 * v;
        float g = y - 0.344 * u - 0.714 * v;
        float b = y + 1.772 * u;
        gl_FragColor = vec4(r, g, b, 1.0);
    }
    '''

    def __init__(self, filename, **kwargs):
        super(CeedDataReader, self).__init__(**kwargs)
        self.filename = filename

    def close_h5(self):
        if self._nix_file is not None:
            self._nix_file.close()
            self._nix_file = None

    def open_h5(self):
        if self._nix_file is not None:
            raise Exception('File already open')

        self._nix_file = nix.File.open(
            self.filename, nix.FileMode.ReadOnly)

    @staticmethod
    def get_128_electrode_names(movie_padding=True):
        names = []
        disabled = set([
            'A1', 'A2', 'A3', 'A10', 'A11', 'A12',
            'B1', 'B2', 'B11', 'B12',
            'C1', 'C12',
            'K1', 'K12',
            'L1', 'L2', 'L11', 'L12',
            'M1', 'M2', 'M3', 'M10', 'M11', 'M12'])
        for row in range(12, 0, -1):
            for col in 'ABCDEFGHJKLM':
                name = '{}{}'.format(col, row)
                names.append(None if name in disabled else name)
        return names

    def get_experiments(self):
        experiments = []
        for block in self._nix_file.blocks:
            m = re.match(self._experiment_pat, block.name)
            if m is not None:
                experiments.append(m.group(1))
        return list(sorted(experiments, key=int))

    def load_experiment(self, experiment):
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

        if ('experiment_{}'.format(experiment) in
                self._nix_file.blocks['ceed_mcs_alignment'].data_arrays):
            self.electrode_intensity_alignment = self._nix_file.blocks[
                'ceed_mcs_alignment'].data_arrays[
                'experiment_{}'.format(experiment)]
        else:
            self.electrode_intensity_alignment = None
            logging.warning(
                'Could not find alignment for experiment {}'.format(experiment))

    def load_mcs_data(self):
        if self.electrodes_data:  # already loaded
            return

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

    @staticmethod
    def _show_image(canvas, pos, size, img, scale, translation, rotation):
        from kivy.graphics.texture import Texture
        from kivy.graphics.fbo import Fbo
        from kivy.graphics import (
            Mesh, StencilPush, StencilUse, StencilUnUse, StencilPop, Rectangle,
            Color)
        from kivy.graphics.context_instructions import (
            PushMatrix, PopMatrix, Rotate, Translate, Scale, MatrixInstruction,
            BindTexture)
        from kivy.graphics.transformation import Matrix
        img_fmt = img.get_pixel_format()
        img_w, img_h = img.get_size()

        if img_fmt not in ('yuv420p', 'rgba', 'rgb24', 'gray', 'bgr24', 'bgra'):
            ofmt = get_best_pix_fmt(
                img_fmt, ('yuv420p', 'rgba', 'rgb24', 'gray', 'bgr24', 'bgra'))
            swscale = SWScale(
                iw=img_w, ih=img_h, ifmt=img_fmt, ow=0, oh=0, ofmt=ofmt)
            img = swscale.scale(img)
            img_fmt = img.get_pixel_format()

        kivy_ofmt = {
            'yuv420p': 'yuv420p', 'rgba': 'rgba', 'rgb24': 'rgb',
            'gray': 'luminance', 'bgr24': 'bgr', 'bgra': 'bgra'}[img_fmt]

        if kivy_ofmt == 'yuv420p':
            w2 = int(img_w / 2)
            h2 = int(img_h / 2)
            tex_y = Texture.create(size=(img_w, img_h), colorfmt='luminance')
            tex_u = Texture.create(size=(w2, h2), colorfmt='luminance')
            tex_v = Texture.create(size=(w2, h2), colorfmt='luminance')

            with canvas:
                fbo = Fbo(size=(img_w, img_h))
            with fbo:
                BindTexture(texture=tex_u, index=1)
                BindTexture(texture=tex_v, index=2)
                Rectangle(size=fbo.size, texture=tex_y)
            fbo.shader.fs = CeedDataReader._YUV_RGB_FS
            fbo['tex_y'] = 0
            fbo['tex_u'] = 1
            fbo['tex_v'] = 2

            tex = fbo.texture
            dy, du, dv, _ = img.to_memoryview()
            tex_y.blit_buffer(dy, colorfmt='luminance')
            tex_u.blit_buffer(du, colorfmt='luminance')
            tex_v.blit_buffer(dv, colorfmt='luminance')
        else:
            tex = Texture.create(size=(img_w, img_h), colorfmt=kivy_ofmt)
            tex.blit_buffer(img.to_memoryview()[0], colorfmt=kivy_ofmt)
        tex.flip_vertical()

        with canvas:
            StencilPush()
            Rectangle(pos=pos, size=size)
            StencilUse()

            PushMatrix()
            center = pos[0] + size[0] / 2, pos[1] + size[1] / 2
            Rotate(angle=rotation, axis=(0, 0, 1), origin=center)
            Scale(scale, scale, 1, origin=center)
            Translate(*translation)
            Rectangle(size=(img_w, img_h), texture=tex, pos=pos)
            PopMatrix()

            StencilUnUse()
            Rectangle(pos=pos, size=size)
            StencilPop()

    @callable_gen
    def paint_electrodes_data(
            self, pos, size, orig_size, fbo, rate, electrodes, cols=1, rows=None,
            spacing=2, draw_size=(0, 0), draw_size_hint=(1, 1),
            draw_pos=(0, 0), draw_pos_hint=(None, None), volt_scale=1e-6,
            time_axis_s=1, volt_axis=100):
        from kivy.graphics import (
            Mesh, StencilPush, StencilUse, StencilUnUse, StencilPop, Rectangle,
            Color)
        if not cols and not rows:
            raise ValueError("Either rows or cols must be specified")
        if rows and cols:
            raise ValueError('Only one of cols and rows can be specified')

        if not rows:
            rows = int(math.ceil(len(electrodes) / cols))
        if not cols:
            cols = int(math.ceil(len(electrodes) / rows))

        orig_w, orig_h = orig_size
        draw_w, draw_h = draw_size
        draw_hint_w, draw_hint_h = draw_size_hint
        w = int(draw_w if draw_hint_w is None else orig_w * draw_hint_w)
        h = int(draw_h if draw_hint_h is None else orig_h * draw_hint_h)

        draw_x, draw_y = draw_pos
        draw_hint_x, draw_hint_y = draw_pos_hint
        x = int(draw_x if draw_hint_x is None else orig_w * draw_hint_x)
        y = int(draw_y if draw_hint_y is None else orig_h * draw_hint_y)

        ew = int((w - max(0, cols - 1) * spacing) / cols)
        eh = int((h - max(0, rows - 1) * spacing) / rows)

        positions = [(0, 0), ] * len(electrodes)
        graphics = [None, ] * len(electrodes)
        i = 0
        fbo.add(Color(1, 215 / 255, 0, 1))
        for row in range(rows):
            if i >= len(electrodes):
                break
            ey = y
            if row:
                ey += (eh + spacing) * row

            for col in range(cols):
                if i >= len(electrodes):
                    break
                if electrodes[i] is None:
                    i += 1
                    continue

                ex = x
                if col:
                    ex += (ew + spacing) * col

                positions[i] = ex, ey
                fbo.add(StencilPush())
                fbo.add(Rectangle(pos=(ex, ey), size=(ew, eh)))
                fbo.add(StencilUse())
                graphics[i] = Mesh(mode='line_strip')
                fbo.add(graphics[i])
                fbo.add(StencilUnUse())
                fbo.add(Rectangle(pos=(ex, ey), size=(ew, eh)))
                fbo.add(StencilPop())

                i += 1
        fbo.add(Color(1, 1, 1, 1))

        electrodes_data = [None, ] * len(electrodes)
        # y_min, y_max = float('inf'), float('-inf')
        alignment = np.array(self.electrode_intensity_alignment)
        for name in electrodes:
            if name is not None:
                break
        freq = self.electrodes_metadata[name]['sampling_frequency']

        frame_n = int(1 / rate * freq)
        n_t = int(time_axis_s * freq)
        t_vals = np.arange(n_t) / (n_t - 1) * ew
        y_scale = (eh / 2) / volt_axis

        for i, name in enumerate(electrodes):
            if name is None:
                continue
            offset, scale = self.get_electrode_offset_scale(name)
            electrodes_data[i] = data = \
                self.electrodes_data[name], offset, scale / volt_scale
            # y_min = min(np.min(data), y_min)
            # y_max = max(np.max(data), y_max)

        k_start = None
        while True:
            i = yield
            if i >= len(alignment):
                raise ValueError('Count value is too large')
            k = alignment[i]

            if k_start is None:
                k_start = k

            k = k + frame_n if i == len(alignment) - 1 else alignment[i + 1]
            n = (k - k_start) % n_t
            t = t_vals[:n]

            for data, (ex, ey), mesh in zip(
                    electrodes_data, positions, graphics):
                if data is None:
                    continue
                data, offset, scale = data

                data = (np.array(data[k - n:k]) - offset) * (y_scale * scale)
                assert len(data) == len(t)

                verts = [0, ] * (4 * len(data))
                for j, (x, y) in enumerate(zip(t, data)):
                    verts[j * 4] = x + ex
                    verts[j * 4 + 1] = y + ey + eh / 2
                mesh.vertices = verts
                mesh.indices = range(len(data))

    def generate_movie(
            self, filename, out_fmt='yuv420p', codec='libx264',
            lib_opts={'crf': '0'}, start=None, end=None, canvas_size=(0, 0),
            canvas_size_hint=(1, 1), shape_pos=(0, 0),
            shape_pos_hint=(None, None), metadata_funcs=[], alpha=1, lum=1,
            speed=1, background=None):
        from kivy.graphics import (
            Canvas, Translate, Fbo, ClearColor, ClearBuffers, Scale)
        from kivy.core.window import Window

        rate = float(self.view_controller.frame_rate)
        rate_int = int(rate)
        if rate != rate_int:
            raise ValueError('Frame rate should be integer')
        orig_w, orig_h = (
            self.view_controller.screen_width,
            self.view_controller.screen_height)

        canvas_w, canvas_h = canvas_size
        cv_hint_w, cv_hint_h = canvas_size_hint
        w = int(canvas_w if cv_hint_w is None else orig_w * cv_hint_w)
        h = int(canvas_h if cv_hint_h is None else orig_h * cv_hint_h)

        shape_x, shape_y = shape_pos
        shape_hint_x, shape_hint_y = shape_pos_hint
        x = int(shape_x if shape_hint_x is None else orig_w * shape_hint_x)
        y = int(shape_y if shape_hint_y is None else orig_h * shape_hint_y)

        Window.size = w, h
        intensities = self.shapes_intensity

        n = len(intensities[next(iter(intensities.keys()))])
        if start is not None:
            start = int(start * rate)
            if start >= n:
                raise Exception('Start time is after the end of the data')
        else:
            start = 0

        if end is not None:
            end = int(math.ceil(end * rate)) + 1
            if end <= start:
                raise Exception('End time is before or at the start time')
        else:
            end = n

        stream = {
            'pix_fmt_in': 'rgba', 'pix_fmt_out': out_fmt,
            'width_in': w, 'height_in': h, 'width_out': w,
            'height_out': h, 'codec': codec,
            'frame_rate': (int(speed * rate_int), 1)}
        writer = MediaWriter(filename, [stream], fmt='mp4', lib_opts=lib_opts)

        fbo = Fbo(size=(w, h), with_stencilbuffer=True)
        with fbo:
            ClearColor(0, 0, 0, 1)
            ClearBuffers()
            Scale(1, -1, 1)
            Translate(0, -h, 0)

        metadata_funcs = [
            f((x, y), (w, h), (orig_w, orig_h), fbo, rate)
            for f in metadata_funcs]

        if background is not None:
            background(
                fbo, (x, y),
                (self.view_controller.screen_width,
                 self.view_controller.screen_height))

        fbo.draw()
        img = Image(plane_buffers=[fbo.pixels], pix_fmt='rgba', size=(w, h))
        writer.write_frame(img, 0.)

        fbo.add(Translate(x, y))
        shape_views = self.stage_factory.get_shapes_gl_color_instructions(
            fbo, 'stage_replay')
        fbo.add(Translate(-x, -y))

        for i in tqdm(range(start, end)):
            for name, intensity in intensities.items():
                r, g, b, a = intensity[i]
                if not r and not g and not b:
                    a = 0
                shape_views[name].rgba = r * lum, g * lum, b * lum, a * alpha
            for f in metadata_funcs:
                f(i)

            fbo.draw()
            img = Image(plane_buffers=[fbo.pixels], pix_fmt='rgba', size=(w, h))
            writer.write_frame(img, (i - start + 1) / (rate * speed))

    @staticmethod
    def populate_config(
            settings, shape_factory, function_factory, stage_factory):
        old_name_to_shape_map = {}
        shape_factory.set_state(settings['shape'], old_name_to_shape_map)

        funcs, func_name_map = function_factory.recover_funcs(
            settings['function'])

        stages, stage_name_map = stage_factory.recover_stages(
            settings['stage'], func_name_map,
            old_name_to_shape_map=old_name_to_shape_map)
        return funcs, stages

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

    def dump_electrode_data_matlab(self, prefix, chunks=1e9):
        itemsize = np.array([0.0]).nbytes
        data = self.electrodes_data
        n_items = None  # num time samples per chunk
        if chunks is not None:
            # number of bytes per sample with all channels
            n_items = int(chunks // (itemsize * len(data)))

        total_n = sum(len(value) for value in data.values())
        pbar = tqdm(
            total=total_n, file=sys.stdout, unit_scale=1, unit='bytes')

        for name, value in data.items():
            pbar.desc = 'Electrode {:6}'.format(name)
            filename = prefix + '_' + name + '.mat'
            with open(filename, 'wb') as fh:
                offset, scale = self.get_electrode_offset_scale(name)

                if chunks is None:
                    items = np.array(value)
                    scipy.io.savemat(fh, {'data': (items - offset) * scale,
                                          'sr': self.electrodes_metadata[name]['sampling_frequency']})
                    pbar.update(len(items))
                else:
                    n = len(value)
                    i = 0
                    while i * n_items < n:
                        items = np.array(
                            value[i * n_items:min((i + 1) * n_items, n)])
                        scipy.io.savemat(fh, {
                            '{}_{}'.format(name, i): (items - offset) * scale})
                        pbar.update(len(items))
                        i += 1
        pbar.close()

    def dump_electrode_data_circus(self, filename, chunks=1e9):
        self.load_mcs_data()
        itemsize = np.array([0.0], dtype=np.float32).nbytes
        data = self.electrodes_data
        n = len(next(iter(data.values())))  # num samples per channel
        n_items = int(chunks // itemsize)  # num chunked samples per chan
        total_n = sum(len(value) for value in data.values())  # num bytes total
        pbar = tqdm(
            total=total_n * itemsize, file=sys.stdout, unit_scale=1,
            unit='bytes')

        mmap_array = open_memmap(
            filename, mode='w+', dtype=np.float32, shape=(n, len(data)))

        names = sorted(data.keys(), key=lambda x: (x[0], int(x[1:])))
        for k, name in enumerate(names):
            value = data[name]
            offset, scale = self.get_electrode_offset_scale(name)
            i = 0
            n = len(value)

            while i * n_items < n:
                items = np.array(
                    value[i * n_items:min((i + 1) * n_items, n)])
                mmap_array[i * n_items:i * n_items + len(items), k] = \
                    (items - offset) * scale
                pbar.update(len(items) * itemsize)
                i += 1
        pbar.close()
        print('Channel order is: {}'.format(names))


if __name__ == '__main__':
    from functools import partial
    f = CeedDataReader(r'/home/cpl/Desktop/data/test_merged.h5')
    f.open_h5()
    f.load_mcs_data()

    # for exp in f.get_experiments():
    #     f.load_experiment(exp)
    #     f.generate_movie(
    #         r'/home/cpl/Desktop/data/test_merged{}.mp4'.format(exp), lum=3, speed=.25)
    # exit()

    f.load_experiment(0)
    background = None
    # img = f.get_fluorescent_image()
    # background = partial(
    #     f._show_image, img=img, scale=f.view_controller.cam_scale,
    #     translation=(
    #         f.view_controller.cam_center_x - img.get_size()[0] / 2,
    #         f.view_controller.cam_center_y - img.get_size()[1] / 2
    #     ),
    #     rotation=f.view_controller.cam_rotation)

    # f.generate_movie(
    #     r'E:\test.mp4', lum=3, speed=.1, background=background)

    f.generate_movie(
        r'/home/cpl/Desktop/data/test_merged_electrodes.mp4', lum=3, canvas_size_hint=(2, 1), speed=.2,
        metadata_funcs=[f.paint_electrodes_data(
            electrodes=CeedDataReader.get_128_electrode_names(),
            draw_pos_hint=(1, 0), volt_axis=50, cols=12)],
        background=background
    )
    f.close_h5()

    # f = CeedDataReader(r'/home/cpl/Desktop/test_out.h5')
    # f.open_h5()
    # f.load_experiment(0)
    # f.save_flourescent_image(r'/home/cpl/Desktop/test_out.bmp')
