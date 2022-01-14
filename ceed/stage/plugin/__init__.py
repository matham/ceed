"""
.. _stage-plugin:

Stage plugins
=====================

Defines a plugin architecture so that new stages can be defined at runtime
and made available to the :class:`~ceed.stage.StageFactoryBase`
used in the GUI to list available stages, and for :mod:`analysis`.

:func:`ceed.stage.register_all_stages` is called by the GUI and it
registers any plugin stages with the
:class:`ceed.stage.StageFactoryBase` used by the GUI for managing
stages. :func:`ceed.stage.register_all_stages` internally calls
:func:`get_plugin_stages` to get all the stages exported by all the
Python plugin module files in the ``ceed/stage/plugin`` directory and
registers them with the :class:`ceed.stage.StageFactoryBase`.

Additionally, if the user provides a package name in the
:attr:`~ceed.main.CeedApp.external_stage_plugin_package` configuration
variable, the GUI will similarly import and register the plugins in
that package with :func:`ceed.stage.register_external_stages`.
The package must however be a proper python package that can be imported.
E.g. if :attr:`~ceed.main.CeedApp.external_stage_plugin_package`
is ``my_ceed_plugin.stage``, Ceed will try something roughly like
``from my_ceed_plugin.stage import get_ceed_stages``.

Files in ``ceed/stage/plugin`` that want to define new stage classes should
define a function in the file called ``get_ceed_stages`` that
returns a list of stages that will be automatically registered with the stage
factory :class:`~ceed.stage.StageFactoryBase` using
:meth:`~ceed.stage.StageFactoryBase.register`.

To write a plugin, familiarize yourself with
:class:`~ceed.stage.CeedStage`, it's properties, and relevant methods that
need to be overridden. Some relevant methods are
:meth:`~ceed.function.FuncBase.get_gui_props`,
:meth:`~ceed.function.FuncBase.get_state`,
:meth:`~ceed.function.FuncBase.get_noise_supported_parameters`,
:meth:`~ceed.function.FuncBase.get_prop_pretty_name`,
:meth:`~ceed.function.FuncBase.init_func_tree`,
:meth:`~ceed.function.FuncBase.init_func`,
:meth:`~ceed.function.FuncBase.init_loop_iteration`,
:meth:`~ceed.function.FuncBase.get_relative_time`, and
:meth:`~ceed.function.FuncBase.resample_parameters`.

:meth:`~ceed.function.CeedFunc.__call__` is how the function is called. It
takes the time in global time so it should convert it to function local time
with :meth:`~ceed.function.FuncBase.get_relative_time` and then return the
function value.

See the ``ceed/stage/plugin/__init__.py`` file and
:func:`get_ceed_stages` for an example plugin (currently the internal plugin
doesn't actually define any new stages).
"""

import importlib
import pathlib
from typing import Iterable, Union, Tuple, List, Type, Dict, Any, Optional
import csv
from textwrap import dedent

from kivy.properties import StringProperty
from ceed.stage import StageType, StageFactoryBase, CeedStage, \
    StageDoneException
from ceed.utils import get_plugin_modules

__all__ = (
    'get_plugin_stages', 'get_ceed_stages', 'CSVStage')


class CSVStage(CeedStage):
    """Defines a stage whose shape frame values are read from a csv file.

    The format of the CSV file is one column for each shape and color channel.
    The top row contains the name of the shape followed by an underscore and the
    color channel. E.g. if the shape is named ``circle`` and we have values for
    its red channel, then the column name will be ``circle_red``.

    The remaining rows, each row in the column is the value of that shape's
    color channel for one frame. The total duration of the stage is the number
    of frames (rows) divided by the frame rate. I.e. each row corresponds to a
    frame, but the depending on the fps in Ceed it can be displayed at
    different times.

    Color channels not provided in the file are not set by the stage. E.g. if
    the file only had one column headed with ``circle_red``, the circle's
    green and blue channels won't be set.

    Values are forced to the ``[0, 1]`` range.
    """

    csv_path: str = StringProperty('')
    """The full path to the CSV file.
    """

    _csv_values: Dict[str, List[List[Optional[float]]]] = {}

    _n_csv_values: int = 0

    def __init__(self, **kwargs):
        self.name = 'CSVStage'
        super().__init__(**kwargs)

    def get_state(self, *args, **kwargs) -> dict:
        s = super(CSVStage, self).get_state(*args, **kwargs)
        s['csv_path'] = self.csv_path
        return s

    def add_func(self, *args, **kwargs) -> None:
        raise TypeError('Cannot add function to CSV stage')

    def add_stage(self, *args, **kwargs) -> None:
        raise TypeError('Cannot add stage to CSV stage')

    def add_shape(self, *args, **kwargs) -> None:
        raise TypeError('Cannot manually add shape to CSV stage')

    def get_settings_display(self, *args, **kwargs) -> Dict[str, Any]:
        items = super(CSVStage, self).get_settings_display(*args, **kwargs)
        from kivy.lang import Builder
        box = Builder.load_string(dedent("""
        BoxLayout:
            spacing: '5dp'
            size_hint_y: None
            height: self.minimum_height
            size_hint_min_x: self.minimum_width
            FlatImageButton:
                scale_down_color: True
                source: 'flat_folder.png'
                hover_text: 'Select CSV file'
                flat_color: app.theme.accent
                on_release: app.open_filechooser(\
callback=root.load_csv, target=text_input.text, title='Select CSV file', \
filters=['*.csv', '*.*'])
            FlatTextInput:
                id: text_input
                size_hint_min_x: '40dp'
                size_hint_y: None
                height: self.minimum_height
                multiline: False
        """))
        text_input = box.ids.text_input

        def load_csv(paths):
            if not paths:
                return
            self.csv_path = paths[0]

        def set_text(*_):
            text_input.text = self.csv_path
            self._csv_values = {}
            self._n_csv_values = 0

        def set_path(*_):
            if not text_input.focus:
                self.csv_path = text_input.text

        self.fbind('csv_path', set_text)
        text_input.fbind('focus', set_path)
        box.load_csv = load_csv

        items['CSV path'] = box
        return items

    def init_stage_tree(self, *args, **kwargs) -> None:
        super().init_stage_tree(*args, **kwargs)
        if not self._csv_values:
            self._read_csv()

    def _read_csv(self):
        existing_shapes = self.shape_factory.shape_names

        with open(self.csv_path) as fp:
            reader = csv.reader(fp)
            names = next(reader)

            shapes: Dict[str, List[Optional[int]]] = {}
            for i, name in enumerate(names):
                if name.endswith('_red'):
                    name = name[:-4]
                    k = 0
                elif name.endswith('_green'):
                    name = name[:-6]
                    k = 1
                elif name.endswith('_blue'):
                    name = name[:-5]
                    k = 2
                else:
                    raise ValueError(
                        f'Column {i} with name {name} does not end in either '
                        f'"_red", "_green", or "_blue"')

                if name not in shapes:
                    if name not in existing_shapes:
                        raise TypeError(
                            f'Unable to find shape "{name}" in the Ceed shapes')
                    shapes[name] = [None, None, None]
                shapes[name][k] = i

            csv_values = self._csv_values = {n: [] for n in shapes}
            self._n_csv_values = 0

            def cond(val):
                return max(0., min(1., float(val)))

            for row in reader:
                self._n_csv_values += 1
                items = list(map(cond, row))
                for name, indices in shapes.items():
                    csv_values[name].append([
                        None if i is None else items[i] for i in indices])

    def tick_stage_loop(self, shapes, last_end_t):
        r, g, b = self.color_r, self.color_g, self.color_b
        a = self.color_a
        csv_values = self._csv_values
        # t is the next time to be used
        t = yield

        for i in range(self._n_csv_values):
            for name, items in csv_values.items():
                r_val, g_val, b_val = items[i]
                values = (r_val if r else None, g_val if g else None,
                          b_val if b else None, a)
                shapes[name].append(values)

            last_end_t = t
            t = yield

        self.t_end = last_end_t
        raise StageDoneException

    def get_stage_shape_names(self):
        if not self._csv_values:
            self._read_csv()
        return set(self._csv_values.keys())


def get_plugin_stages(
        stage_factory: StageFactoryBase, base_package: str,
        root: Union[str, pathlib.Path]
) -> Tuple[List[Type[StageType]], List[Tuple[Tuple[str], bytes]]]:
    """Imports all the ``.py`` files in the given directory and sub-directories
    for the named package that don't start with a underscore (except for
    ``__init__.py`` of course, which is imported). For each imported module,
    it calls its ``get_ceed_stages``
    function (if they are defined in the module) which should return a list
    (that can be empty) of all the stages classes exported by the module.

    It then returns all these exported stages and all the file contents in the
    folder.

    Ceed will automatically import all the plugins under
    ``ceed/stage/plugin``.

    :parameter stage_factory: The
        :class:`~ceed.stage.StageFactoryBase` instance with which the
        returned plugin stages will be registered.
    :parameter base_package: The package name from which the plugins will be
        imported. E.g. to import the plugins in ``ceed/stage/plugin``,
        ``base_package`` is ``ceed.stage.plugin`` because that's the package
        containing name for the ``plugin`` directory. Then, they are imported
        as ``ceed.stage.plugin`` and ``ceed.stage.plugin.xyz``, if
        the plugin directory also contains a ``xyz.py`` plugin file.
    :parameter root: The full directory path that contains the plugins. E.g.
        for ``ceed.stage.plugin`` it is ``ceed/stage/plugin``.
    :returns:
        A tuple with two values containing: a list of stage classes
        exported by all the modules and a list containing all the python files
        contents encountered in the directory.

        The python files contents are returned so that Ceed can store it in the
        data files in case the plugins are changed between experimental runs.

        See :func:`get_ceed_stages` and :func:`~ceed.stage.register_all_stages`
        for an example how it's used.
    """
    stages = []
    packages, files = get_plugin_modules(base_package, root)

    for package in packages:
        m = importlib.import_module(package)
        if hasattr(m, 'get_ceed_stages'):
            stages.extend(m.get_ceed_stages(stage_factory))

    return stages, files


def get_ceed_stages(
        stage_factory: StageFactoryBase) -> Iterable[Type[StageType]]:
    """Returns all the stage classes defined and exported in this file
    (currently none for the internal plugin).

    :param stage_factory: The :class:`~ceed.function.FunctionFactoryBase`
        instance currently active in Ceed.
    """
    return (CSVStage, )
