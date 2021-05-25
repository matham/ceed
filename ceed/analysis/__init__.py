"""
.. _ceed-analysis:

Ceed data analysis
==================

Reads a combined Ceed and MCS H5 data file that has been merged with
:mod:`~ceed.analysis.merge_data`, using :class:`CeedDataReader`.

During an experiment Ceed saves its projector data to a NIX H5 file and MCS
saves its electrode data to a proprietary file. After the experiment, one
exports the MCS electrode data to an H5 file through the MCS software. Then,
following :mod:`~ceed.analysis.merge_data`, one creates a new Ceed based file
that also contains the electrode data as well as an alignment between the Ceed
video frames and their temporal location in the MCS data.
:class:`CeedDataReader` can be used to load this combined file.

See the examples directory in the Ceed Github repository and
:ref:`example-analysis` for complete worked out examples using the data.

Loading data
------------

:class:`CeedDataReader` is used to read the MCS electrode data and Ceed data.
A typical example is::

    # this internally opens the file and ensures it's closed when the with
    # block exits, even if there was an exception
    with CeedDataReader('data.h5') as f:
        # load the electrode data
        f.load_mcs_data()

        print('Found experiments: {}. Number of saved images are: {}'.format(
            f.experiments_in_file, f.num_images_in_file))

        # load a specific experiment
        f.load_experiment(0)
        # do something with the experiment data. E.g. save the image
        f.save_image('image.bmp', f.experiment_cam_image)

The overall steps are to create the :class:`CeedDataReader` and enter its
``with`` block. This internally calls :meth:`CeedDataReader.open_h5`
and :meth:`CeedDataReader.close_h5`.

Next, you can load the MCS electrode
data and metadata using :meth:`CeedDataReader.load_mcs_data`. This loads it
into memory as sets the instance properties with the data (see
:attr:`CeedDataReader.electrodes_data` and
:attr:`CeedDataReader.electrodes_metadata`).

Finally, you can load the Ceed data as well as the Ceed-MCS alignment for a
specific experiment using :meth:`CeedDataReader.load_experiment`. This
sets further class properties and allows you to use instance methods that
require the Ceed data. E.g. see :attr:`CeedDataReader.shapes_intensity` and
:attr:`CeedDataReader.electrode_intensity_alignment`.

Experiment movie
----------------

:class:`CeedDataReader` can create a movie that replays the experiment
and the intensity of the shapes. It can also draw the tissue image,
MEA electrode grid, and the electrode voltage.

See :meth:`CeedDataReader.generate_movie` for details. E.g.::

    with CeedDataReader('data.h5') as f:
        f.load_mcs_data()
        f.load_experiment(0)
        f.generate_movie(
            'video.mp4', lum=3, canvas_size_hint=(2, 1),
            speed=1., end=1,
            paint_funcs=[
                f.paint_background_image(
                    f.experiment_cam_image,
                    transform_matrix=f.view_controller.cam_transform),
                f.show_mea_outline(f.view_controller.mea_transform),
                f.paint_electrodes_data_callbacks(
                    f.get_electrode_names(), draw_pos_hint=(1, 0),
                    volt_axis=50)]
        )

This creates a movie of the first second of the experiment at three times
the intensity, containing the background image, the MEA grid and the electrode
data as well as the shapes.
"""

import math
import scipy.io
import numpy as np
import sys
from fractions import Fraction
import itertools
from functools import wraps
import nixio as nix
import pathlib
from typing import List, Union, Optional, Dict, Any, Tuple
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

__all__ = (
    'CeedDataReader', 'EndOfDataException', 'CallableGen', 'callable_gen',
    'partial_func', 'read_nix_prop')


class CallableGen:
    """Converts a generator that accepts input into a callable.

    It calls ``next`` on the generator when created. Them with each call it
    "sends" the argument when called to the generator.

    E.g.::

        >>> def echo():
        ...     while True:
        ...         value = yield
        ...         print(f'Got "{value}"')
        >>> # use it like a normal generator
        >>> generator = echo()
        >>> next(generator)
        >>> generator.send(1)
        Got "1"
        >>> generator.send(5)
        Got "5"
        >>> # or more conveniently make it into a callable
        >>> callback_generator = CallableGen(echo())
        >>> callback_generator(1)
        Got "1"
        >>> callback_generator(5)
        Got "5"
    """

    def __init__(self, gen):
        self.gen = gen
        next(gen)

    def __call__(self, i):
        self.gen.send(i)

    def close(self):
        """Closes the underlying generator.
        """
        if self.gen is not None:
            self.gen.close()
        self.gen = None


def callable_gen(gen):
    """Decorator that converts a generator into a :class:`CallableGen`.

    It returns a function that when called will return a
    function that itself instantiates the original decorated generator and
    passes it to :class:`CallableGen`, that is returned.

    Both of these functions accept positional and keyword arguments that are
    passed on to the generator when it's created.
    """
    @wraps(gen)
    def outer_func(*l, **kw):
        @wraps(gen)
        def inner_func(*largs, **kwargs):
            return CallableGen(gen(*l, *largs, **kwargs, **kw))
        return inner_func
    return outer_func


def partial_func(func):
    """Decorator that returns a function, which when called returns another
    function that calls the original function.

    Both of these functions accept positional and keyword arguments that are
    passed on to the original function when it's called.
    """
    @wraps(func)
    def outer_func(*l, **kw):
        @wraps(func)
        def inner_func(*largs, **kwargs):
            return func(*l, *largs, **kwargs, **kw)
        return inner_func
    return outer_func


def read_nix_prop(prop):
    """Reads the value from a nix property.

    It is slightly different between versions, so this works across all nix
    versions.
    """
    try:
        return prop.values[0].value
    except AttributeError:
        return prop.values[0]


class EndOfDataException(Exception):
    """Exception raised when the data read is after the end of the file.
    """
    pass


class CeedDataReader:
    """Loads the data from an experiment.

    See :mod:`~ceed.analysis` for details.
    """

    filename: str = ''
    """The full filename path of the Ceed H5 file opened.
    """

    experiments_in_file: List[str] = []
    """After :meth:`open_h5`, it is a list of the experiment names
    found in the file. The experiments listed can be opened with
    :meth:`load_experiment`.
    """

    num_images_in_file: int = 0
    """After :meth:`open_h5`, it is the number of camera images
    found in the file. Images can be opened by number with
    :meth:`get_image_from_file`.
    """

    electrodes_data: Dict[str, np.ndarray] = {}
    """After :meth:`load_mcs_data`, it is a mapping whose keys is an
    electrode name and whose values is a 1D array with the raw electrode data
    from MCS.

    Use :meth:`get_electrode_offset_scale` to convert the raw data to properly
    scaled volt.

    The electrode data is typically sampled at a high sampling rate and contains
    multiple Ceed experiments. See :attr:`electrode_intensity_alignment`.
    """

    electrodes_metadata: Dict[str, Dict[str, Any]] = {}
    """Similar to :attr:`electrodes_data`, but instead the values is a dict of
    the electrode metadata such as sampling rate etc.
    """

    electrode_dig_data: Optional[np.ndarray] = None
    """After :meth:`load_mcs_data`, if present in the file, it's an 1D array
    containing the digital input data as sampled by the MCS system.

    This data was used when aligning the Ceed data to the MCS electrode data in
    :mod:`~ceed.analysis.merge_data` and it contains the alignment pattern sent
    by Ceed to MCS over the hardware link. There should be the same number of
    samples as in the :attr:`electrodes_data` for any electrode.
    """

    electrode_intensity_alignment: Optional[np.ndarray] = None
    """After :meth:`load_experiment` it is a 1D array, mapping the ceed
    frame data (:attr:`shapes_intensity_rendered`) to the MCS
    :attr:`electrodes_data` by index.

    Ceed refreshes the projector at a low rate, e.g. 120Hz. MCS records data at
    a very high rate, e.g. 10kHz. Consequently, each Ceed frame is displayed
    during multiple electrode samples. So, starting from the first frame for an
    experiment, we can write the it starts at index ``k`` in
    :attr:`electrodes_data`, frame 2 at index ``m``, frame 3 at index ``n``,
    etc. Then e.g. frame one is displayed during samples ``[k, m)``, frame two
    at ``[m, n)``, etc.

    :attr:`electrode_intensity_alignment` are these start indices, one for each
    Ceed frame and therefore its length is the number of Ceed frames displayed
    during the experiment.

    Minor detail; if the experiment ran in QUAD4X or QUAD12X mode then each main
    frame is composed of sub-frames (e.g. 4 or 12 frames). The start index of
    each sub-frame was estimated by splitting the main frames into the given
    number of sub-frames. Because we only know the exact index of each main
    frame.

    As explained in :ref:`dropped-frames`, Ceed will sometimes drop a frame to
    compensate when a frame is displayed too long. Therefore,
    the length of :attr:`electrode_intensity_alignment` may be shorter than the
    the length of the values in :attr:`shapes_intensity`, because
    this contains only indices for frames that were rendered. So it should be
    the same length as the items in :attr:`shapes_intensity_rendered`.

    If Ceed drops a frame, it means a previous frame was displayed for longer
    than one frame duration. Consequently, the distance between values in this
    array may not be regularly spaced. E.g. in the example above, ``m - n``
    could be twice or three times as large as ``k - m``.
    :meth:`CeedDataReader.compute_long_and_skipped_frames` helps analyze to find
    these dropped and long frames.
    """

    shapes_intensity: Dict[str, np.ndarray] = {}
    """After :meth:`load_experiment`, it is a mapping whose keys are the
    names of the shapes drawn in ceed for this experiment and whose values
    is a 2D array containing the intensity value of the shape for each Ceed
    frame.

    This is sampled at the projector frame rate, not the MEA sampling rate and
    should have the same size or larger as the shape of
    :attr:`electrode_intensity_alignment`.

    Each shape is a 2D array. Each row corresponds to a single frame and has 4
    columns indicating the red, green, blue, and alpha intensity of that shape
    for that frame. They are values between 0-1. Alpha is the transparency with
    1 being completely opaque.

    As explained in :attr:`electrode_intensity_alignment`, Ceed will drop frames
    to compensate for long frames. However, Ceed will still compute and record
    the shapes' intensity for dropped frames. :attr:`shapes_intensity` contains
    the intensity for all frames, including the dropped ones and may therefore
    be larger than the length of :attr:`electrode_intensity_alignment`.

    :attr:`shapes_intensity_rendered` contains only the frames that were
    rendered and is therefore the same size as
    :attr:`electrode_intensity_alignment`. It is generated by indexing
    :attr:`shapes_intensity` using :attr:`rendered_frames`.

    :attr:`rendered_frames_long` additionally indicates the displayed frames
    that were displayed for more than one frame duration, potentially causing
    Ceed to drop frames.
    """

    shapes_intensity_rendered: Dict[str, np.ndarray] = {}
    """After :meth:`load_experiment`, it is a mapping whose keys are the
    names of the shapes drawn in ceed for this experiment and whose values
    is a 2D array containing the intensity value of the shape for each Ceed
    frame.

    Unlike :attr:`shapes_intensity`, it only contains the frames that were
    displayed and is the same size as :attr:`electrode_intensity_alignment`.
    See those properties docs' for details.
    """

    rendered_frames: np.ndarray = None
    """After :meth:`load_experiment`, it is a 1D logical array of the same size
    as the values in :attr:`shapes_intensity` and it indicates
    which frames of those frames were rendered.

    :attr:`shapes_intensity_rendered` is internally generated by indexing
    :attr:`shapes_intensity` with :attr:`rendered_frames`.

    If MCS was stopped recording data while a Ceed experiment was ongoing,
    those remaining frames are marked as not rendered because they are not in
    the MCS data, even though they may actually have been projected.
    """

    rendered_frames_long: Optional[np.ndarray] = None
    """After :meth:`load_experiment` it is a 1D logical array of the same size
    as :attr:`shapes_intensity_rendered` indicating which of the rendered frames
    were displayed for longer than one frame. See :attr:`shapes_intensity`.

    It's None when :attr:`electrode_intensity_alignment` is None. If there are
    overwhelmingly too many long frames it may not be accurate.
    """

    led_state: np.ndarray = None
    """After :meth:`load_experiment`, it is the state of each of the 3 RGB
    projector LEDs.

    The projector lets us set it them using
    :attr:`~ceed.view.controller.ViewControllerBase.LED_mode` and this records
    it when set during the experiment.

    It is a 2D array of Tx4:

    * where each row corresponds to each time the projector is set during the
      experiment,
    * the first column is the frame number when it was set as indexed into
      the values in :attr:`shapes_intensity`,
    * the remaining 3 columns is the state of the R, G, and B LED as set by
      the projector.
    """

    view_controller: ViewControllerBase = None
    """After :meth:`load_experiment`, it is the
    :class:`~ceed.view.controller.ViewControllerBase` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    data_serializer: DataSerializerBase = None
    """After :meth:`load_experiment`, it is the
    :class:`~ceed.storage.controller.DataSerializerBase` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    function_factory: FunctionFactoryBase = None
    """After :meth:`load_experiment`, it is the
    :class:`~ceed.function.FunctionFactoryBase` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    stage_factory: StageFactoryBase = None
    """After :meth:`load_experiment`, it is the
    :class:`~ceed.stage.StageFactoryBase` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    shape_factory: CeedPaintCanvasBehavior = None
    """After :meth:`load_experiment`, it is the
    :class:`~ceed.shape.CeedPaintCanvasBehavior` instance configured to
    the same settings as used during the :attr:`loaded_experiment`.
    """

    experiment_stage_name: str = ''
    """After :meth:`load_experiment`, it is the name of the stage among
    the stages in :attr:`stage_factory` that was used to run the currently
    :attr:`loaded_experiment`.
    """

    experiment_notes: str = ''
    """The experiment notes recorded for the currently
    :attr:`loaded_experiment` in the GUI.
    """

    experiment_start_time: float = 0.
    """The ``time.time`` timestamp when the currently
    :attr:`loaded_experiment` started.
    """

    experiment_cam_image: Optional[Image] = None
    """The camera image stored for the currently :attr:`loaded_experiment`,
    if there was a background image present in the GUI when it was started.

    See also :meth:`save_image`.
    """

    loaded_experiment: Optional[str] = None
    """After :meth:`load_experiment`, it is the experiment number of the
    currently loaded experiment.

    The ceed h5 file contains data for multiple experiments.
    :meth:`load_experiment` must be called to load a particular experiment;
    this stores the experiment number currently loaded.
    """

    ceed_version: str = ''
    """After :meth:`open_h5`, it is the version of ceed that created
    the file.
    """

    external_function_plugin_package: str = ''
    """After :meth:`open_h5`, it contains the external function plugin
    package containing any additional functions, if it was specified during the
    experiment.

    See :attr:`~ceed.main.CeedApp.external_function_plugin_package`.
    """

    external_stage_plugin_package: str = ''
    """After :meth:`open_h5`, it contains the external stage plugin
    package containing any additional stages, if it was specified during the
    experiment.

    See :attr:`~ceed.main.CeedApp.external_stage_plugin_package`
    """

    app_logs: str = ''
    """After :meth:`open_h5`, it contains the overall app logs recorded during
    the experiments.
    """

    app_notes: str = ''
    """After :meth:`open_h5`, it contains the overall notes stored in the file,
    including those added during
    :meth:`~ceed.analysis.merge_data.CeedMCSDataMerger.merge_data`.
    """

    app_config: Dict[str, Any] = {}
    """After :meth:`open_h5`, it is the last ceed application
    configuration objects as saved in the file.

    These options are similarly stored in each experiment containing the
    configuration options used at the time the experiment was executed. These
    are stored in :attr:`function_factory`, :attr:`stage_factory` etc. and
    is specific to the experiment. :attr:`app_config` are the Ceed options as
    they were when the file was last closed/saved.
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

        This is implicitly called when using the ``with`` syntax shown in
        :meth:`open_h5`, which is a safer way of closing the file because it
        is called even if there's an exception.
        """
        if self._nix_file is not None:
            self._nix_file.close()
            self._nix_file = None

    def open_h5(self):
        """Opens the data file in read only mode and loads the application
        config.

        This is implicitly called when using the ``with`` syntax below, which is
        a safer way of opening the file because :meth:`close_h5` is called even
        if there's an exception.

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

        This will call :meth:`open_h5` when entring the ``with`` block and
        :meth:`close_h5` when exiting, even if there's an exception raised
        within the block.
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
        """After :meth:`load_experiment`, it returns a 2D list (list of lists)
        containing the 2D grid of electrode names used in the experiment
        as derived from the
        :attr:`~ceed.view.controller.ViewControllerBase.mea_num_cols` and other
        mea options.
        """
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
        """Loads the data for a specific experiment and populates the instance
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
        n_sub_frames = 1
        if view.video_mode == 'QUAD4X':
            n_sub_frames = 4
        elif view.video_mode == 'QUAD12X':
            n_sub_frames = 12

        self.experiment_cam_image = self.read_image_from_block(
            self._block)

        rendered_counter = np.asarray(block.data_arrays['frame_time_counter'])
        frame_counter_n = block.data_arrays['frame_counter'].shape[0]

        count_indices = np.arange(1, 1 + frame_counter_n)
        found = rendered_counter[:, np.newaxis] - \
            np.arange(n_sub_frames)[np.newaxis, :]
        found = found.reshape(-1)
        rendered_frames = np.isin(count_indices, found)
        self.rendered_frames = rendered_frames

        if ('ceed_mcs_alignment' in self._nix_file.blocks and
                'experiment_{}'.format(experiment) in
                self._nix_file.blocks['ceed_mcs_alignment'].data_arrays):
            alignment = self.electrode_intensity_alignment = np.asarray(
                self._nix_file.blocks[
                    'ceed_mcs_alignment'].data_arrays[
                    'experiment_{}'.format(experiment)])
            long_frames = self.rendered_frames_long = np.empty(
                len(alignment), dtype=np.bool)

            if len(alignment) >= 2:
                diff = alignment[1:] - alignment[:-1]
                long_frames[:-1] = diff >= 1.5 * np.median(diff)
                long_frames[-1] = 0
            else:
                long_frames[:] = 0

            # it's possible that the alignment is shorter than number of
            # rendered frames if MCS was stopped in middle of recording. So Ceed
            # would have recorded frames but not MCS. So chop it OFF
            rendered_i = np.arange(len(rendered_frames))[rendered_frames]
            if len(rendered_i) > len(alignment):
                # frames larger than alignments size were not recorded
                rendered_frames[rendered_i[len(alignment)]:] = False
        else:
            self.electrode_intensity_alignment = None
            self.rendered_frames_long = None

        data = self.shapes_intensity = {}
        data_rendered = self.shapes_intensity_rendered = {}
        for item in block.data_arrays:
            if not item.name.startswith('shape_'):
                continue
            name = item.name[6:]
            values = data[name] = np.asarray(item)
            data_rendered[name] = values[rendered_frames, :]

        self.led_state = np.asarray(block.data_arrays['led_state'])

    def load_mcs_data(self):
        """Loads the MCS electrode data and stores them in the instance
        properties.

        See e.g. :attr:`electrodes_data`.

        It should be called only once after opening the file. E.g.::

            >>> with CeedDataReader(filename='data.h5') as reader:
            ...     reader.load_mcs_data()
            ...     reader.load_experiment(0)
            ...     print(reader.electrodes_metadata)
            ...     print(reader.experiment_notes)
        """
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

    def get_image_from_file(self, image_num: int) -> Tuple[Image, str, float]:
        """After opening the file :attr:`num_images_in_file` contains the
        number of images the user manually saved to the file. This loads
        a specific image and its metadata.

        See also :meth:`save_image`.

        :param image_num: The image number to return. Should be less than
            :attr:`num_images_in_file`.
        :return: A 3-tuple of ``(image, notes, save_time)``, where ``image`` is
            the image, ``notes`` is any user notes saved with the image, and
            ``save_time`` is the ``time.time`` when the image was saved.
        """
        block = self._nix_file.blocks['fluorescent_images']
        postfix = '_{}'.format(image_num)

        group = block.groups['fluorescent_image{}'.format(postfix)]
        img = self.read_image_from_block(block, postfix)

        notes = group.metadata['notes']
        save_time = float(group.metadata['save_time'])

        return img, notes, save_time

    def save_image(self, filename: str, img: Image, codec: str = 'bmp'):
        """Saves the given image to disk.

        :param filename: The full path where to save the image.
        :param img: The image to save.
        :param codec: The codec with which to save. The default is bmp and the
            extension should then also be ``.bmp``.
        """
        BaseRecorder.save_image(
            filename, img, codec=codec, pix_fmt=img.get_pixel_format())

    def get_electrode_offset_scale(self, electrode: str) -> Tuple[float, float]:
        """After :meth:`load_mcs_data`, returns the conversion factors used
        to convert the raw integer data in :attr:`electrodes_data` to properly
        scaled micro-volt.

        :param electrode: The electrode name as stored in
            :attr:`electrodes_data`
        :return: A 2-tuple of `(offset, scale) `` that can be used to convert.
            E.g. ``micro-volt = (input - offset) * scale``.
        """
        metadata = self.electrodes_metadata[electrode]
        return (
            float(metadata['ADZero']),
            float(metadata['ConversionFactor']) *
            10. ** float(metadata['Exponent']))

    def estimate_skipped_frames(self):
        """After :meth:`load_experiment`, returns an estimate of the skipped
        and long frames.

        It returns the same value as :attr:`compute_long_and_skipped_frames`,
        but specific to the current experiment.
        """
        n_sub_frames = 1
        if self.view_controller.video_mode == 'QUAD4X':
            n_sub_frames = 4
        elif self.view_controller.video_mode == 'QUAD12X':
            n_sub_frames = 12
        return self.compute_long_and_skipped_frames(
            n_sub_frames, self.rendered_frames,
            self.electrode_intensity_alignment)

    @staticmethod
    def compute_long_and_skipped_frames(
            n_sub_frames: int, rendered_frames: np.ndarray,
            ceed_mcs_alignment: np.ndarray
    ) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Returns a estimate of the Ceed frames that were displayed longer
        than one frame and the frames that were dropped.

        ``ceed_mcs_alignment`` can be shorter than ``rendered_frames`` because
        the MCS data could have been stopped recording while Ceed was still
        projecting.

        :param n_sub_frames: The number of sub-frames for each Ceed frame (e.g.
            quad mode can be 4 or 12).
        :param rendered_frames: The logical array indexing the rendered frames.
            It is the same as :attr:`rendered_frames`.
        :param ceed_mcs_alignment: The Ceed-MCS alignment array.
            It is the same as :attr:`electrode_intensity_alignment`.
        :return: A 5-tuple of ``(mcs_long_frames, mcs_frame_len, ceed_skipped,
            ceed_skipped_main, largest_bad)``. ``mcs_long_frames`` is the
            frame indices that were longer than one. ``mcs_frame_len`` is the
            corresponding length of each long frame. ``ceed_skipped`` is the
            indices of the frames that were skipped. ``ceed_skipped_main`` is
            the indices of the *main* frames that were skipped. ``largest_bad``
            is the maximum delay in terms of frames between Ceed repeating a
            frame because the CPU was too slow, until Ceed realized it needs to
            drop a frame to compensate. I.e. this is the largest number of bad
            frames Ceed ever displayed before correcting for each event.
        """
        ceed_not_rendered = np.logical_not(rendered_frames)
        # indices of ceed dropped frames, including sub-frames
        ceed_skipped_i = np.arange(len(ceed_not_rendered))[ceed_not_rendered]

        ceed_not_rendered = ceed_not_rendered[::n_sub_frames]
        ceed_frame_indices = np.arange(len(ceed_not_rendered))
        # indices of main ceed dropped frames
        ceed_skipped_main_i = ceed_frame_indices[ceed_not_rendered]

        alignment = ceed_mcs_alignment[::n_sub_frames]
        diff = alignment[1:] - alignment[:-1]
        mcs_frame_len = np.round(diff / np.median(diff))
        long_frames = mcs_frame_len > 1
        rendered = ceed_frame_indices[np.logical_not(ceed_not_rendered)][:-1]
        # get all long frames, cap to MCS frames if ceed is longer
        mcs_long_frames = rendered[:len(long_frames)][long_frames]
        mcs_frame_len = mcs_frame_len[long_frames]
        mcs_frame_len = mcs_frame_len.astype(np.int32)

        num_long = sum(i - 1 for i in mcs_frame_len)

        if num_long and num_long == len(ceed_skipped_main_i):
            flat_mcs_long = [
                f + k for f, duration in zip(mcs_long_frames, mcs_frame_len)
                for k, i in enumerate(range(duration - 1))
            ]
            largest_bad = max(
                c - m for m, c in zip(flat_mcs_long, ceed_skipped_main_i))
        elif num_long:
            largest_bad = 'unmatched'
        else:
            largest_bad = 0

        return (
            mcs_long_frames, mcs_frame_len, ceed_skipped_i, ceed_skipped_main_i,
            largest_bad)

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
            self, img: Image, scale=None, translation=None,
            rotation=None, transform_matrix=None):
        """Uses :func:`partial_func` to return a function that can be passed to
        ``paint_funcs`` of :meth:`generate_movie` which will draw the image
        in the background.

        This can be used e.g. to display the tissue image and display the
        experiment shapes on it.

        :param img: The image to show.
        :param scale: Optionally a scaling factor by which to upscale the image.
        :param translation: Optionally a 2-tuple of a x, y translation by which
            to translate the image.
        :param rotation: Optionally a rotation angle in degrees by which to
            rotate the image.
        :param transform_matrix: Optionally a transformation matrix stored as
            an array by which to apply a affine transform to the image.

            You can get the transformation used for the current experiment
            from :attr:`~ceed.view.controller.ViewControllerBase.cam_transform`,
            using :attr:`view_controller`.
        :return: The function to pass to :meth:`generate_movie`.

        E.g.::

            >>> with CeedDataReader('data.h5') as f:
            ... f.load_mcs_data()
            ... f.load_experiment(0)
            ... paint_funcs = [
            ...     f.paint_background_image(
            ...         f.experiment_cam_image,
            ...         transform_matrix=f.view_controller.cam_transform)
            ... ]
            ... f.generate_movie('video.mp4', paint_funcs=paint_funcs)
        """
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
        """Uses :func:`partial_func` to return a function that can be passed to
        ``paint_funcs`` of :meth:`generate_movie` which will draw the
        :attr:`electrodes_data`.

        This can be used e.g. to display the electrode data for all electrodes
        while the movie is displaying the intensity of each of the shapes.

        :param electrode_names: List of names of electrodes (as keys in
            :attr:`electrodes_data`) to draw.
        :param spacing: The pixel spacing by which to space them to each other.
        :param draw_size: A optional 2-tuple of the size of the overall grid
            in which to draw the electrodes.
        :param draw_size_hint: Similar to ``draw_size``, but instead of absolute
            size, it's the scaling factor relative to the projector screen size
            on which the Ceed shapes are drawn. E.g. ``(1, 1)`` means it'll be
            the same size as the shapes region, and ``(1, 2)`` means it'll be
            the same horizontally, but twice as large vertically.
        :param draw_pos: A optional 2-tuple of the bottom left corner pos of the
            overall grid in which to draw the electrodes.
        :param draw_pos_hint: Similar to ``draw_pos``, but instead of absolute
            pos, it's the scaling factor relative to the projector screen size
            on which the Ceed shapes are drawn. E.g. ``(1, 0)`` means it'll be
            drawn to the right of the shapes region but at the same vertical
            height.
        :param volt_scale: The scale in which the volts are given. E.g. 1e-6
            means that the data is given in micro-volts (which is what MCS
            defaults to).
        :param time_axis_s: The sub-sampling rate of the x-axis in seconds.
            E.g. 1 means that we display 1 sample for every second.
        :param volt_axis: The plus/minus range of the y-axis to display, in
            units of ``volt_scale``. E.g. 100 means it'll display ``+/- 100uv``.
        :return: The function to pass to :meth:`generate_movie`.

        E.g.::

            >>> with CeedDataReader('data.h5') as f:
            ... f.load_mcs_data()
            ... f.load_experiment(0)
            ... paint_funcs = [
            ...     f.paint_electrodes_data_callbacks(
            ...     f.get_electrode_names(), draw_pos_hint=(1, 0),
            ...     volt_axis=50)
            ... ]
            ... f.generate_movie(
            ...     'video.mp4',
            ...     canvas_size_hint=(2, 1), paint_funcs=paint_funcs)
        """
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
        """Uses :func:`partial_func` to return a function that can be passed to
        ``paint_funcs`` of :meth:`generate_movie` which will draw the MEA
        electrode grid in the movie.

        This can be used e.g. to display the position of the electrodes
        relative to the tissue and relative to the experiment shapes.

        :param transform_matrix: Optionally a transformation matrix stored as
            an array by which to apply a affine transform to the grid.

            You can get the transformation used for the current experiment
            from :attr:`~ceed.view.controller.ViewControllerBase.mea_transform`,
            using :attr:`view_controller`.
        :param color: A 4-tuple of the red, green, blue, and transparency
            value to use to draw the grid.
        :return: The function to pass to :meth:`generate_movie`.

        E.g.::

            >>> with CeedDataReader('data.h5') as f:
            ... f.load_mcs_data()
            ... f.load_experiment(0)
            ... paint_funcs = [
            ...     f.show_mea_outline(f.view_controller.mea_transform)
            ... ]
            ... f.generate_movie('video.mp4', paint_funcs=paint_funcs)
        """
        return partial_func(self._show_mea_outline)(
            transform_matrix=transform_matrix, color=color)

    def generate_movie(
            self, filename, out_fmt='yuv420p', codec='libx264',
            lib_opts={'crf': '0'}, video_fmt='mp4', start=None, end=None,
            canvas_size=(0, 0),
            canvas_size_hint=(1, 1), projector_pos=(0, 0),
            projector_pos_hint=(None, None), paint_funcs=(),
            stimulation_transparency=1., lum=1., speed=1., hidden_shapes=None):
        """Generates a movie from the Ceed and MCS data.

        By default it just draws the Ceed shapes and their intensities for each
        frame. Additional things can be drawn in the movie using e.g.
        :meth:`paint_background_image`,
        :meth:`paint_electrodes_data_callbacks`, and :meth:`show_mea_outline`.

        :param filename: The video filename to create.
        :param out_fmt: The pixel format to use for the movie. Defaults to
            ``yuv420p``
        :param codec: The codec to use for the movie. Defaults to x264.
        :param lib_opts: Any options to pass to video encoding library.
        :param video_fmt: The video format into which to save the video.
            Defaults to mp4.
        :param start: The optional start time, in seconds in terms of the
            experiment, from which to start drawing the movie.
        :param end: The optional end time, in seconds in terms of the
            experiment, at which to end drawing the movie.
        :param canvas_size: The optional movie size to create. If not provided
            defaults to the projector screen size used during the experiment.
        :param canvas_size_hint: Similar to ``canvas_size``, but is relative to
            the projector screen size used during the experiment. e.g.
            ``(1, 2)`` means that it'll be the same width the projector, but
            twice as tall (e.g. if we want to draw other stuff besides the
            shapes in the empty space).
        :param projector_pos: The optional absolute bottom left position where
            to draw the shapes. Defaults to the bottom left corner.
        :param projector_pos_hint: The optional relative bottom left position
            where to draw the shapes. E.g. ``(1, 0)`` means to draw at a
            horizontal offset the size of the projector, but at the zero
            vertical offset.
        :param paint_funcs: Additional functions that the drawing engine will
            call for each Ceed frame so it will be given the opportunity to
            draw other things.

            The function is called once before anything so it can set up. Then
            if it returns a non-None object, it returns a generator that
            we call for each frame using ``gen.send``, where we send it the
            current frame number and it draws the data for that frame. See
            the sample methods for an example.

            The order determines the z-depth of the canvas. E.g. the last
            function will be painted on top of the previous functions.
        :param stimulation_transparency: The transparency used for the shapes.
            1 means completely opaque, zero means completely transparent.
            It determines whether we can see through the drawn shapes to see
            the underlying image.
        :param lum: The luminosity, as a percentage with which to draw the
            shape intensities. E.g. lum of 0.5 means all the shapes will be
            drawn at %50 of the real intensity.
        :param speed: The speed of the movie relative to the rela experiment.
            E.g. 1 means real time. 0.5 means it'll be hals as fast as
            real-time. And 2 means it'll be twice as fast.
        :param hidden_shapes: Optional list of Ceed shape names that will not be
            displayed in the movie.

        E.g.::

            >>> with CeedDataReader('data.h5') as f:
            ... f.load_mcs_data()
            ... f.load_experiment(0)
            ... paint_funcs = [
            ...     f.paint_background_image(
            ...         f.experiment_cam_image,
            ...         transform_matrix=f.view_controller.cam_transform),
            ...     f.show_mea_outline(f.view_controller.mea_transform),
            ...     f.paint_electrodes_data_callbacks(
            ...     f.get_electrode_names(), draw_pos_hint=(1, 0),
            ...     volt_axis=50)
            ... ]
            ... f.generate_movie(
            ...     'video.mp4',
            ...     canvas_size_hint=(2, 1), paint_funcs=paint_funcs)

        .. note::

            The movie generation can be quite slow.
        """
        from kivy.graphics import (
            Canvas, Translate, Fbo, ClearColor, ClearBuffers, Scale)
        from kivy.core.window import Window

        rate = self.view_controller.effective_frame_rate
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

        speed_frac = Fraction(
            *float(speed).as_integer_ratio()).limit_denominator(10_000)
        play_rate = rate * speed_frac
        stream = {
            'pix_fmt_in': 'rgba', 'pix_fmt_out': out_fmt,
            'width_in': w, 'height_in': h, 'width_out': w,
            'height_out': h, 'codec': codec,
            'frame_rate': (play_rate.numerator, play_rate.denominator)}
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
            total=float((end - 1 - start) / rate),
            file=sys.stdout, unit='second', unit_scale=1)

        # all shapes listed in intensities must be in shape_views. However,
        # we don't want to show shapes not given values in intensities or if
        # they are to be hidden
        unused_shapes = set(shape_views) - set(intensities)
        unused_shapes.update(set(hidden_shapes or []))
        for name in unused_shapes:
            if name in shape_views:
                shape_views[name].rgba = 0, 0, 0, 0

        for i in range(start, end):
            pbar.update(float(1 / rate))
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
            writer.write_frame(img, float((i - start + 1) / play_rate))
        pbar.close()

    @staticmethod
    def populate_config(
            settings, shape_factory, function_factory, stage_factory):
        """Given the application settings recovered from e.g. a yaml file
        and the shape, function, and stage factories, it creates the
        shapes, functions, and stages from the settings and returns the
        root functions and stages.
        """
        old_name_to_shape_map = {}
        shape_factory.set_state(settings['shape'], old_name_to_shape_map)

        funcs, func_name_map = function_factory.recover_funcs(
            settings['function'])

        stages, stage_name_map = stage_factory.recover_stages(
            settings['stage'], func_name_map,
            old_name_to_shape_map=old_name_to_shape_map)
        return funcs, stages

    @staticmethod
    def read_image_from_block(block, postfix='') -> Image:
        """Given a nix block that contains an Image, it reads and returns the
        image.
        """
        try:
            group = block.groups['fluorescent_image{}'.format(postfix)]
        except KeyError:
            return None

        planes = [np.array(d).tobytes() for d in group.data_arrays]
        img = Image(plane_buffers=planes, pix_fmt=group.metadata['pix_fmt'],
                    size=yaml_loads(group.metadata['size']))
        return img

    def dump_electrode_data_matlab(self, prefix, chunks=1e9):
        """Dumps all the electrode data in a format that can be loaded
        by matlab. Each electrode is stored in its own file.

        :param prefix: A path to a directory and filename prefix that will be
            used for the output files. The output files, one for each electrode,
            will have the electrode name and a .mat suffix.
        :param chunks: If not None, the data will be split into chucks of the
            given size in bytes and these chucks will be saved in the file.
            If the data is too big we may not be able to load it all at once
            in memory, so saving it as chucks will work if the chucks are small
            enough. Defaults to 1GB.
        """
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
        """Dumps all the electrode data in a format that can be loaded
        by spyking circus.

        :param filename: The filename where to dump the data.
        :param chunks: If not None, the data will be copied in chucks of the
            given size in bytes. If the data is too big we may not be able to
            load it all at once in memory, so saving it with chucks will work
            if the chucks are small enough. Defaults to 1GB.
        """
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
    with CeedDataReader('data.h5') as f:
        f.load_mcs_data()

        print('Found experiments: {}. Number of saved images are: {}'.format(
            f.experiments_in_file, f.num_images_in_file))

        f.load_experiment(0)
        f.save_image('image.bmp', f.experiment_cam_image)

        f.generate_movie(
            'video.mp4', lum=3, canvas_size_hint=(2, 1),
            speed=1., end=1,
            paint_funcs=[
                f.paint_background_image(
                    f.experiment_cam_image,
                    transform_matrix=f.view_controller.cam_transform),
                f.show_mea_outline(f.view_controller.mea_transform),
                f.paint_electrodes_data_callbacks(
                    f.get_electrode_names(), draw_pos_hint=(1, 0),
                    volt_axis=50)]
        )
