"""Ceed data analysis
======================

Code to read the load the recorded Ceed experimental data.

"""

import math
import scipy.io
import numpy as np
import sys
import itertools
import nixio as nix
import pathlib
from typing import List, Union, Optional, Dict, Any
from numpy.lib.format import open_memmap

from more_kivy_app.utils import yaml_loads
from more_kivy_app.config import apply_config
from cpl_media.recorder import BaseRecorder
from ffpyplayer.pic import Image, SWScale
from ffpyplayer.tools import get_best_pix_fmt
from tqdm import tqdm
from ffpyplayer.writer import MediaWriter

from ceed.function import FunctionFactoryBase, register_all_functions, \
    register_external_functions
from ceed.stage import StageFactoryBase, register_external_stages, \
    register_all_stages
from ceed.shape import CeedPaintCanvasBehavior
from ceed.storage.controller import DataSerializerBase, CeedDataWriterBase
from ceed.view.controller import ViewControllerBase

__all__ = ('CeedDataReader', 'EndOfDataException', 'CallableGen',
           'read_nix_prop')


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


def partial_func(func):
    def outer_func(*l, **kw):
        def inner_func(*largs, **kwargs):
            return func(*l, *largs, **kwargs, **kw)
        return inner_func
    return outer_func


def read_nix_prop(prop):
    try:
        return prop.values[0].value
    except AttributeError:
        return prop.values[0]


class EndOfDataException(Exception):
    """
    Exception raised when the data read is after the end of the file.
    """
    pass


class CeedDataReader:

    filename: str = ''
    """The full filename path associated with this :class:`CeedDataReader`
    instance.
    """

    experiments_in_file: List[str] = []
    """Once set in :meth:`open_h5`, it is a list of the experiment names
    found in the file. The experiments listed can be opened with
    :meth:`load_experiment`.
    """

    num_images_in_file: int = 0
    """Once set in :meth:`open_h5`, it is the number of camera images
    found in the file. Images can be opened by number with
    :meth:`get_image_from_file`.
    """

    electrodes_data: Dict[str, np.ndarray] = {}
    """Once set in :meth:`load_mcs_data`, it is a mapping whose keys are
    electrode names and whose values are 1D arrays with the electrode data.
    """

    electrodes_metadata: Dict[str, Dict[str, Any]] = {}
    """Similar to :attr:`electrodes_data`, but instead the values are the
    electrode metadata such as sampling rate etc.
    """

    electrode_dig_data: Optional[np.ndarray] = None
    """If present, it's an array containing the digital input data as sampled
    by the MCS system. There should be the same number of samples in the time
    dimension as for the :attr:`electrodes_data`.
    """

    electrode_intensity_alignment: Optional[np.ndarray] = None
    """Once set in :meth:`load_experiment` it is a 1D array, mapping the ceed
    array data into the MCS :attr:`electrodes_data` by indices.
    """

    shapes_intensity: Dict[str, np.ndarray] = {}
    """Once set in :meth:`load_experiment`, it is a mapping whose keys are the
    names of the shapes available in ceed for this experiment and whose values
    is a 1D array with the intensity value of the shape for each time point.

    This is sampled at the projector frame rate, not the MEA sampling rate and
    should have the same size as the shape of
    :attr:`electrode_intensity_alignment`.
    """

    led_state: np.ndarray = None
    """The projector lets us set the state of each of the 3 RGB LEDs
    independently. This is 2D array of Tx4:

    * whose first dimensions is the number of times the LED state was set by
      the projector,
    * the first column is the frame number when it was set as indexed into
      :attr:`shapes_intensity` for each shape,
    * the remaining 3 columns is the state of the R, G, and B LED as set by
      the projector.
    """

    view_controller: ViewControllerBase = None
    """Once set in :meth:`load_experiment`, it is the
    :class:`~ceed.view.controller.ViewControllerBase` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    data_serializer: DataSerializerBase = None
    """Once set in :meth:`load_experiment`, it is the
    :class:`~ceed.storage.controller.DataSerializerBase` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    function_factory: FunctionFactoryBase = None
    """Once set in :meth:`load_experiment`, it is the
    :class:`~ceed.function.FunctionFactoryBase` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    stage_factory: StageFactoryBase = None
    """Once set in :meth:`load_experiment`, it is the
    :class:`~ceed.stage.StageFactoryBase` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    shape_factory: CeedPaintCanvasBehavior = None
    """Once set in :meth:`load_experiment`, it is the
    :class:`~ceed.shape.CeedPaintCanvasBehavior` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    experiment_stage_name: str = ''
    """Once set in :meth:`load_experiment`, it is the name of the stage among
    the stages in :attr:`stage_factory` that was used to run the currently
    :attr:`loaded_experiment`.
    """

    experiment_notes = ''
    """The text notes from the currently :attr:`loaded_experiment`.
    """

    experiment_start_time = 0
    """The timestamp that the currently :attr:`loaded_experiment` started.
    """

    experiment_cam_image: Optional[Image] = None
    """The camera image stored for the currently :attr:`loaded_experiment`.
    """

    loaded_experiment: Optional[str] = None
    """The ceed h5 file contains multiple experiments and their data.
    :meth:`load_experiment` must be called to load a particular experiment.
    This stores the experiment number currently loaded.
    """

    ceed_version: str = ''
    """Once set in :meth:`open_h5`, it is the version of ceed that created
    the file.
    """

    external_function_plugin_package: str = ''
    """Once set by :meth:`open_h5`, it contains the external function plugin
    package containing any additional functions, if it was specified during the
    experiment.
    """

    external_stage_plugin_package: str = ''
    """Once set by :meth:`open_h5`, it contains the external stage plugin
    package containing any additional stages, if it was specified during the
    experiment.
    """

    app_logs: str = ''
    """Once set in :meth:`open_h5`, it contains the experimental logs
    of ceed for the experiments stored in the file.
    """

    app_notes: str = ''
    """Once set in :meth:`open_h5`, it contains the overall notes stored
    with the file.
    """

    app_config: Dict[str, Any] = {}
    """Set by :meth:`open_h5`; it is the last ceed application
    configuration objects as saved in the file.

    A copy of the options is also stored in each experiment - with the
    configuration options used at the time the experiment was executed. These
    are stored in :attr:`function_factory`, :attr:`stage_factory` etc.
    :attr:`app_config` are these options when the file was last closed/saved
    and it's stored in a dict.
    """

    _nix_file: Optional[nix.File] = None
    """Nix file handle opened with :meth:`open_h5`.
    """

    _block = None

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

    def __enter__(self):
        self.open_h5()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_h5()

    def close_h5(self):
        """Closes the data file if it has been opened.
        """
        if self._nix_file is not None:
            self._nix_file.close()
            self._nix_file = None

    def open_h5(self):
        """Opens the data file in read only mode and loads the application
        config.

        To load a specific experiment data after the file is open, use
        :meth:`load_experiment`.

        E.g.::

            >>> reader = CeedDataReader(filename='data.h5')
            >>> reader.open_h5()
            >>> print(reader.experiments_in_file)
            >>> reader.load_experiment(0)
            >>> reader.close_h5()

        Or in a safer way::

            >>> with CeedDataReader(filename='data.h5') as reader:
            ...     print(reader.experiments_in_file)
            ...     reader.load_experiment(0)
        """
        from ceed.storage.controller import CeedDataWriterBase
        if self._nix_file is not None:
            raise Exception('File already open')

        self._nix_file = nix.File.open(
            self.filename, nix.FileMode.ReadOnly)
        self.experiments_in_file = \
            CeedDataWriterBase.get_blocks_experiment_numbers(
                self._nix_file.blocks)
        self.num_images_in_file = \
            CeedDataWriterBase.get_file_num_fluorescent_images(self._nix_file)
        self.load_application_data()

    def dump_plugin_sources(
            self, plugin_type: str, target_root: Union[str, pathlib.Path]):
        """Dumps the source code of all the registered function or stage
        plugins (depending on ``plugin_type``) that was saved to the data file.

        See :func:`ceed.function.register_external_functions`,
        :func:`ceed.function.register_all_functions`,
        :func:`ceed.stage.register_external_stages`, and
        :func:`ceed.stage.register_all_stages`.

        :param plugin_type: either ``'function'`` or ``'stage'`` to indicate
            which plugin to dump.
        :param target_root: the root directory where to dump the source code.
        """
        root = pathlib.Path(target_root)
        nix_file = self._nix_file

        contents_s = nix_file.sections[
            f'{plugin_type}_plugin_sources']['contents']

        for package, contents in yaml_loads(contents_s).items():
            for file_parts, content in contents:
                src_filename = root.joinpath(package, *file_parts)
                if src_filename.exists():
                    raise ValueError(f'{src_filename} already exists')

                if not src_filename.parent.exists():
                    src_filename.parent.mkdir(parents=True)
                src_filename.write_bytes(content)

    def get_electrode_names(self) -> List[List[str]]:
        names = []
        disabled = {
            'A1', 'A2', 'A3', 'A10', 'A11', 'A12',
            'B1', 'B2', 'B11', 'B12',
            'C1', 'C12',
            'K1', 'K12',
            'L1', 'L2', 'L11', 'L12',
            'M1', 'M2', 'M3', 'M10', 'M11', 'M12'}

        for row in range(1, self.view_controller.mea_num_rows + 1):
            row_names = []
            names.append(row_names)
            for col in "ABCDEFGHJKLMNOPQRSTUVWXYZ"[
                       :self.view_controller.mea_num_cols]:
                name = '{}{}'.format(col, row)
                row_names.append(None if name in disabled else name)
        return names

    def load_application_data(self):
        """Loads all the application configuration from the data file and
        stores it in the instance properties.

        This is automatically called by :meth:`open_h5`.
        """
        self.app_logs = self.app_notes = ''
        if 'app_logs' in self._nix_file.sections:
            self.app_logs = self._nix_file.sections['app_logs']['log_data']
            self.app_notes = self._nix_file.sections['app_logs']['notes']

        config = self._nix_file.sections['app_config']

        config_data = {}
        for prop in config.props:
            config_data[prop.name] = yaml_loads(read_nix_prop(prop))

        self.ceed_version = config_data.get('ceed_version', '')
        self.external_function_plugin_package = config_data.get(
            'external_function_plugin_package', '')
        self.external_stage_plugin_package = config_data.get(
            'external_stage_plugin_package', '')

        view = ViewControllerBase()
        ser = DataSerializerBase()

        func = FunctionFactoryBase()
        register_all_functions(func)
        if self.external_function_plugin_package:
            register_external_functions(
                func, self.external_function_plugin_package)

        shape = CeedPaintCanvasBehavior()
        stage = StageFactoryBase(
            function_factory=func, shape_factory=shape)
        register_all_stages(stage)
        if self.external_stage_plugin_package:
            register_external_stages(
                stage, self.external_stage_plugin_package)

        for name, obj in {
                'view': view, 'serializer': ser, 'function': func}.items():
            if name in config_data['app_settings']:
                apply_config(obj, config_data['app_settings'][name])
        self.populate_config(config_data, shape, func, stage)

        self.app_config = {
            'view_controller': view,
            'data_serializer': ser,
            'function_factory': func,
            'shape_factory': shape,
            'stage_factory': stage,
        }

    def load_experiment(self, experiment: Union[int, str]):
        """Loads the data from a specific experiment and populates the instance
        properties with the data.

        If a previous experiment was loaded, it replaces the properties with
        the new data. E.g.::

            >>> with CeedDataReader(filename='data.h5') as reader:
            ...     print(reader.experiments_in_file)
            ...     reader.load_experiment(0)
            ...     print(reader.experiment_notes)
            ...     reader.load_experiment(2)
        """
        experiment = str(experiment)
        self._block = block = self._nix_file.blocks[
            CeedDataWriterBase.get_experiment_block_name(experiment)]
        section = self._nix_file.sections[
            'experiment{}_metadata'.format(experiment)]
        self.loaded_experiment = experiment

        self.experiment_stage_name = section['stage']
        self.experiment_notes = section['notes'] if 'notes' in section else ''
        self.experiment_start_time = float(
            section['save_time']) if 'save_time' in section else 0
        config = section.sections['app_config']

        config_data = {}
        for prop in config.props:
            config_data[prop.name] = yaml_loads(read_nix_prop(prop))

        view = self.view_controller = ViewControllerBase()
        ser = self.data_serializer = DataSerializerBase()

        func = self.function_factory = FunctionFactoryBase()
        register_all_functions(func)
        package = config_data.get('external_function_plugin_package', '')
        if package:
            register_external_functions(func, package)

        shape = self.shape_factory = CeedPaintCanvasBehavior()
        stage = self.stage_factory = StageFactoryBase(
            function_factory=func, shape_factory=shape)
        register_all_stages(stage)
        package = config_data.get('external_stage_plugin_package', '')
        if package:
            register_external_stages(stage, package)

        for name, obj in {
                'view': view, 'serializer': ser, 'function': func}.items():
            if name in config_data['app_settings']:
                apply_config(obj, config_data['app_settings'][name])
        self.populate_config(config_data, shape, func, stage)

        self.experiment_cam_image = self.read_image_from_block(
            self._block)

        data = self.shapes_intensity = {}
        for item in block.data_arrays:
            if not item.name.startswith('shape_'):
                continue
            data[item.name[6:]] = np.asarray(item)

        self.led_state = np.asarray(block.data_arrays['led_state'])

        if ('ceed_mcs_alignment' in self._nix_file.blocks and
                'experiment_{}'.format(experiment) in
                self._nix_file.blocks['ceed_mcs_alignment'].data_arrays):
            self.electrode_intensity_alignment = np.asarray(
                self._nix_file.blocks[
                    'ceed_mcs_alignment'].data_arrays[
                    'experiment_{}'.format(experiment)])
        else:
            self.electrode_intensity_alignment = None

    def load_mcs_data(self):
        if self.electrodes_data:  # already loaded
            return

        if 'mcs_data' not in self._nix_file.blocks:
            raise TypeError(
                'Cannot load MCS data because no MCS data was merged in file')

        mcs_block = self._nix_file.blocks['mcs_data']
        mcs_metadata = mcs_block.metadata
        self.electrode_dig_data = None
        if 'digital_io' in mcs_block.data_arrays:
            self.electrode_dig_data = np.asarray(
                mcs_block.data_arrays['digital_io'])

        electrode_data = self.electrodes_data = {}
        electrodes_metadata = self.electrodes_metadata = {}
        for item in mcs_block.data_arrays:
            if not item.name.startswith('electrode_'):
                continue
            electrode_data[item.name[10:]] = np.asarray(item)

            electrodes_metadata[item.name[10:]] = electrode_metadata = {}
            for prop in mcs_metadata.sections[item.name].props:
                electrode_metadata[prop.name] = yaml_loads(read_nix_prop(prop))

    def get_image_from_file(self, image_num):
        block = self._nix_file.blocks['fluorescent_images']
        postfix = '_{}'.format(image_num)

        group = block.groups['fluorescent_image{}'.format(postfix)]
        img = self.read_image_from_block(block, postfix)

        notes = group.metadata['notes']
        save_time = float(group.metadata['save_time'])

        return img, notes, save_time

    def save_image(self, filename, img, codec='bmp'):
        BaseRecorder.save_image(
            filename, img, codec=codec, pix_fmt=img.get_pixel_format())

    def get_electrode_offset_scale(self, electrode):
        metadata = self.electrodes_metadata[electrode]
        return (
            float(metadata['ADZero']),
            float(metadata['ConversionFactor']) *
            10. ** float(metadata['Exponent']))

    @staticmethod
    def _show_image(
            config, img, scale=None, translation=None,
            rotation=None, transform_matrix=None):
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
        size = config['orig_size']
        pos = config['pos']
        canvas = config['canvas']

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
            if rotation:
                Rotate(angle=rotation, axis=(0, 0, 1), origin=center)
            if scale:
                Scale(scale, scale, 1, origin=center)
            if translation:
                Translate(*translation)
            if transform_matrix is not None:
                mat = Matrix()
                mat.set(array=transform_matrix)
                m = MatrixInstruction()
                m.matrix = mat
            Rectangle(size=(img_w, img_h), texture=tex, pos=pos)
            PopMatrix()

            StencilUnUse()
            Rectangle(pos=pos, size=size)
            StencilPop()

    def paint_background_image(
            self, img, scale=None, translation=None,
            rotation=None, transform_matrix=None):
        return partial_func(self._show_image)(
            img=img, scale=scale, translation=translation, rotation=rotation,
            transform_matrix=transform_matrix)

    @partial_func
    def _paint_electrodes_data_setup(
            self, config, electrode_names,
            spacing=2, draw_size=(0, 0), draw_size_hint=(1, 1),
            draw_pos=(0, 0), draw_pos_hint=(None, None), volt_scale=1e-6,
            time_axis_s=1, volt_axis=100, transform_matrix=None,
            label_width=70):
        from kivy.graphics import (
            Mesh, StencilPush, StencilUse, StencilUnUse, StencilPop, Rectangle,
            Color)
        from kivy.graphics.context_instructions import (
            PushMatrix, PopMatrix, Scale, MatrixInstruction)
        from kivy.graphics.transformation import Matrix
        from kivy.base import EventLoop
        EventLoop.ensure_window()
        from kivy.core.text import Label
        from kivy.metrics import dp, sp

        n_rows = len(electrode_names)
        if not n_rows:
            raise ValueError("There must be at least one electrode specified")
        n_cols = len(electrode_names[0])
        if not n_cols:
            raise ValueError("There must be at least one electrode specified")
        if not all((len(row) == n_cols for row in electrode_names)):
            raise ValueError(
                "The number of electrodes in all rows must be the same")
        n_electrodes = sum(map(len, electrode_names))

        orig_w, orig_h = config['orig_size']
        fbo = config['canvas']

        label_height = 45 if label_width else 0
        draw_w, draw_h = draw_size
        draw_hint_w, draw_hint_h = draw_size_hint
        w = int(draw_w if draw_hint_w is None else orig_w * draw_hint_w)
        h = int(draw_h if draw_hint_h is None else orig_h * draw_hint_h)

        draw_x, draw_y = draw_pos
        draw_hint_x, draw_hint_y = draw_pos_hint
        x = int(draw_x if draw_hint_x is None else orig_w * draw_hint_x)
        y = int(draw_y if draw_hint_y is None else orig_h * draw_hint_y)

        ew = int((w - label_width - max(0, n_cols - 1) * spacing) / n_cols)
        eh = int((h - label_height - max(0, n_rows - 1) * spacing) / n_rows)

        with fbo:
            PushMatrix()
            # center = x + w / 2., y + h / 2.
            # if scale:
            #     Scale(scale, scale, 1, origin=center)
            if transform_matrix is not None:
                mat = Matrix()
                mat.set(array=transform_matrix)
                m = MatrixInstruction()
                m.matrix = mat

        positions = [(0, 0), ] * n_electrodes
        graphics = [None, ] * n_electrodes
        i = 0

        electrode_color = 1, 215 / 255, 0, 1
        for row, row_names in enumerate(reversed(electrode_names)):
            ey = y + label_height
            if row:
                ey += (eh + spacing) * row

            for col, name in enumerate(row_names):
                if name is None:
                    i += 1
                    continue

                ex = x + label_width
                if col:
                    ex += (ew + spacing) * col

                positions[i] = ex, ey
                fbo.add(Color(*electrode_color))
                fbo.add(StencilPush())
                fbo.add(Rectangle(pos=(ex, ey), size=(ew, eh)))
                fbo.add(StencilUse())
                graphics[i] = Mesh(mode='line_strip')
                fbo.add(graphics[i])
                fbo.add(StencilUnUse())
                fbo.add(Rectangle(pos=(ex, ey), size=(ew, eh)))
                fbo.add(StencilPop())

                i += 1

                if label_width:
                    if not col:
                        fbo.add(Color(1, 1, 1, 1))
                        label = Label(text=name, font_size=sp(40))
                        label.refresh()
                        _w, _h = label.texture.size
                        rect = Rectangle(
                            pos=(x, ey + (eh - _h) / 2.),
                            size=label.texture.size)
                        rect.texture = label.texture
                        fbo.add(rect)

                    if not row:
                        fbo.add(Color(1, 1, 1, 1))
                        label = Label(text=name, font_size=sp(40))
                        label.refresh()
                        _w, _h = label.texture.size
                        rect = Rectangle(
                            pos=(ex + (ew - _w) / 2., y),
                            size=label.texture.size)
                        rect.texture = label.texture
                        fbo.add(rect)

        with fbo:
            Color(1, 1, 1, 1)
            PopMatrix()

        electrodes_data = [None, ] * n_electrodes
        # y_min, y_max = float('inf'), float('-inf')
        alignment = np.array(self.electrode_intensity_alignment)

        # get the frequency from any channel
        name = None
        for row_names in electrode_names:
            for name in row_names:
                if name is not None:
                    break
            if name is not None:
                break
        freq = self.electrodes_metadata[name]['sampling_frequency']

        frame_n = int(1 / config['rate'] * freq)
        n_t = int(time_axis_s * freq)
        t_vals = np.arange(n_t) / (n_t - 1) * ew
        y_scale = (eh / 2) / volt_axis

        for i, name in enumerate(itertools.chain(*electrode_names)):
            if name is None:
                continue
            offset, scale = self.get_electrode_offset_scale(name)
            electrodes_data[i] = \
                self.electrodes_data[name], offset, scale / volt_scale
            # y_min = min(np.min(data), y_min)
            # y_max = max(np.max(data), y_max)

        new_config = {
            'alignment': alignment, 'frame_n': frame_n, 't_vals': t_vals,
            'y_scale': y_scale, 'electrodes_data': electrodes_data, 'n_t': n_t,
            'positions': positions, 'graphics': graphics, 'eh': eh}
        return CallableGen(self._paint_electrodes_data(new_config))

    def _paint_electrodes_data(self, config):
        alignment = config['alignment']
        frame_n = config['frame_n']
        t_vals = config['t_vals']
        y_scale = config['y_scale']
        electrodes_data = config['electrodes_data']
        n_t = config['n_t']
        positions = config['positions']
        graphics = config['graphics']
        eh = config['eh']

        indices = list(range(len(t_vals)))
        verts = np.zeros(len(t_vals) * 4)
        k_start = None
        while True:
            i = yield
            if i >= len(alignment):
                raise EndOfDataException('Count value is too large')
            k = alignment[i]

            if k_start is None:
                k_start = k

            k = k + frame_n if i == len(alignment) - 1 else alignment[i + 1]
            n = (k - k_start) % n_t

            for data, (ex, ey), mesh in zip(
                    electrodes_data, positions, graphics):
                if data is None:
                    continue

                x_vals = t_vals[:n] + ex

                data, offset, scale = data
                data = (np.array(data[k - n:k]) - offset) * (y_scale * scale)
                y_vals = data + ey + eh / 2

                assert len(y_vals) == len(x_vals)

                verts[:4 * n:4] = x_vals
                verts[1:4 * n:4] = y_vals
                mesh.vertices = memoryview(np.asarray(verts, dtype=np.float32))
                mesh.indices = indices[:n]

    def paint_electrodes_data_callbacks(
            self, electrode_names, spacing=2, draw_size=(0, 0),
            draw_size_hint=(1, 1), draw_pos=(0, 0), draw_pos_hint=(None, None),
            volt_scale=1e-6, time_axis_s=1, volt_axis=100):
        return self._paint_electrodes_data_setup(
            electrode_names=electrode_names, spacing=spacing,
            draw_size=draw_size, draw_size_hint=draw_size_hint,
            draw_pos=draw_pos, draw_pos_hint=draw_pos_hint,
            volt_scale=volt_scale, time_axis_s=time_axis_s, volt_axis=volt_axis)

    def _show_mea_outline(
            self, config, transform_matrix=None,
            color=(1, 215 / 255, 0, .2)):
        from kivy.graphics import (
            Line, StencilPush, StencilUse, StencilUnUse, StencilPop, Rectangle,
            Color)
        from kivy.graphics.context_instructions import (
            PushMatrix, PopMatrix, Rotate, Translate, Scale, MatrixInstruction,
            BindTexture)
        from kivy.graphics.transformation import Matrix
        from kivy.base import EventLoop
        EventLoop.ensure_window()
        from kivy.core.text import Label
        from kivy.metrics import dp, sp

        size = config['orig_size']
        pos = config['pos']
        canvas = config['canvas']
        mea_w = max(self.view_controller.mea_num_cols - 1, 0) * \
            self.view_controller.mea_pitch
        mea_h = max(self.view_controller.mea_num_rows - 1, 0) * \
            self.view_controller.mea_pitch
        last_col = "ABCDEFGHJKLMNOPQRSTUVWXYZ"[
            self.view_controller.mea_num_cols - 1]

        with canvas:
            StencilPush()
            Rectangle(pos=pos, size=size)
            StencilUse()

            PushMatrix()
            if transform_matrix is not None:
                mat = Matrix()
                mat.set(array=transform_matrix)
                m = MatrixInstruction()
                m.matrix = mat
            Color(*color)
            Line(points=[0, 0, mea_w, 0, mea_w, mea_h, 0, mea_h], close=True)

            label = Label(text='A1', font_size=sp(12))
            label.refresh()
            _w, _h = label.texture.size
            rect = Rectangle(
                pos=(mea_w, mea_h - _h / 2.), size=label.texture.size)
            rect.texture = label.texture

            label = Label(
                text='A{}'.format(self.view_controller.mea_num_rows),
                font_size=sp(12))
            label.refresh()
            _w, _h = label.texture.size
            rect = Rectangle(
                pos=(-_w, mea_h - _h / 2.), size=label.texture.size)
            rect.texture = label.texture

            label = Label(text='{}1'.format(last_col), font_size=sp(12))
            label.refresh()
            _w, _h = label.texture.size
            rect = Rectangle(pos=(mea_w, -_h / 2.), size=label.texture.size)
            rect.texture = label.texture

            label = Label(
                text='{}{}'.format(last_col, self.view_controller.mea_num_rows),
                font_size=sp(12))
            label.refresh()
            _w, _h = label.texture.size
            rect = Rectangle(pos=(-_w, -_h / 2.), size=label.texture.size)
            rect.texture = label.texture
            PopMatrix()

            StencilUnUse()
            Rectangle(pos=pos, size=size)
            StencilPop()

    def show_mea_outline(
            self, transform_matrix=None, color=(1, 215 / 255, 0, .2)):
        return partial_func(self._show_mea_outline)(
            transform_matrix=transform_matrix, color=color)

    def generate_movie(
            self, filename, out_fmt='yuv420p', codec='libx264',
            lib_opts={'crf': '0'}, video_fmt='mp4', start=None, end=None,
            canvas_size=(0, 0),
            canvas_size_hint=(1, 1), projector_pos=(0, 0),
            projector_pos_hint=(None, None), paint_funcs=(),
            stimulation_transparency=1., lum=1., speed=1., hidden_shapes=None):
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

        projector_x, projector_y = projector_pos
        projector_hint_x, projector_hint_y = projector_pos_hint
        x = int(projector_x if projector_hint_x is None else
                orig_w * projector_hint_x)
        y = int(projector_y if projector_hint_y is None else
                orig_h * projector_hint_y)

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
        writer = MediaWriter(
            filename, [stream], fmt=video_fmt, lib_opts=lib_opts)

        fbo = Fbo(size=(w, h), with_stencilbuffer=True)
        with fbo:
            ClearColor(0, 0, 0, 1)
            ClearBuffers()
            Scale(1, -1, 1)
            Translate(0, -h, 0)

        config = {'canvas': fbo, 'pos': (x, y), 'size': (w, h),
                  'orig_size': (orig_w, orig_h), 'rate': rate}
        paint_funcs = [func(config) for func in paint_funcs]
        paint_funcs = [func for func in paint_funcs if func is not None]

        fbo.draw()
        img = Image(plane_buffers=[fbo.pixels], pix_fmt='rgba', size=(w, h))
        writer.write_frame(img, 0.)

        fbo.add(Translate(x, y))
        shape_views = self.stage_factory.get_shapes_gl_color_instructions(
            fbo, 'stage_replay')
        fbo.add(Translate(-x, -y))

        pbar = tqdm(
            total=(end - 1 - start) / rate, file=sys.stdout, unit='second',
            unit_scale=1)

        # all shapes listed in intensities must be in shape_views. However,
        # we don't want to show shapes not given values in intensities or if
        # they are to be hidden
        unused_shapes = set(shape_views) - set(intensities)
        unused_shapes.update(set(hidden_shapes or []))
        for name in unused_shapes:
            if name in shape_views:
                shape_views[name].rgba = 0, 0, 0, 0

        for i in range(start, end):
            pbar.update(1 / rate)
            for name, intensity in intensities.items():
                r, g, b, a = intensity[i]
                if name in unused_shapes:
                    a = 0
                shape_views[name].rgba = \
                    r * lum, g * lum, b * lum, a * stimulation_transparency

            try:
                for func in paint_funcs:
                    func(i)
            except EndOfDataException:
                break

            fbo.draw()
            img = Image(plane_buffers=[fbo.pixels], pix_fmt='rgba', size=(w, h))
            writer.write_frame(img, (i - start + 1) / (rate * speed))
        pbar.close()

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
    def read_image_from_block(block, postfix=''):
        try:
            group = block.groups['fluorescent_image{}'.format(postfix)]
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
                    scipy.io.savemat(
                        fh, {'data': (items - offset) * scale,
                             'sr': self.electrodes_metadata[name][
                                 'sampling_frequency']}
                    )
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
        print('Channel order in "{}" is: {}'.format(filename, names))


if __name__ == '__main__':
    f = CeedDataReader(r'e:\ceed_merged.h5')
    f.open_h5()
    f.load_mcs_data()

    print('Found experiments: {}. Number of saved images are: {}'.format(
        f.experiments_in_file, f.num_images_in_file))

    f.load_experiment(2)
    # f.save_image(
    #     r'/home/cpl/Desktop/test_out.bmp', f.experiment_cam_image)

    f.generate_movie(
        r'E:\est_merged_electrodes2.mp4', lum=3, canvas_size_hint=(2, 1),
        speed=1., end=1,
        paint_funcs=[
            f.paint_background_image(
                f.experiment_cam_image,
                transform_matrix=f.view_controller.cam_transform),
            f.show_mea_outline(f.view_controller.mea_transform),
            f.paint_electrodes_data_callbacks(
                f.get_electrode_names(), draw_pos_hint=(1, 0), volt_axis=50)]
    )
    f.close_h5()
