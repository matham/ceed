"""Stage plugins
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
from typing import Iterable, Union, Tuple, List, Type

from ceed.stage import StageType, StageFactoryBase
from ceed.utils import get_plugin_modules

__all__ = (
    'get_plugin_stages', 'get_ceed_stages')


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
    return ()
