from ceed.shape import CeedPaintCanvasBehavior, CeedPaintCircle, \
    CeedPaintEllipse, CeedPaintPolygon
from ceed.function import FunctionFactoryBase, register_all_functions, \
    register_external_functions
from ceed.function.plugin import ConstFunc
from ceed.stage import StageFactoryBase, register_all_stages, \
    register_external_stages, CeedStage
from ceed.storage.controller import CeedDataWriterBase
from typing import Type

# create shape/function/stage factory
shape_factory = CeedPaintCanvasBehavior()
function_factory = FunctionFactoryBase()
stage_factory = StageFactoryBase(
    function_factory=function_factory, shape_factory=shape_factory)

# now we need to register all the builtin and external functions and stage
# plugins
register_all_functions(function_factory)
register_all_stages(stage_factory)
# if you have plugins in an external packages, you need to register it as well.
# See the functions for details and how to make the plugin available in the GUI
# register_external_functions(function_factory, 'my_package.function')
# register_external_stages(stage_factory, 'my_package.stage')

# create the shapes and name them. We use the Shape classes imported from Ceed
ellipse = CeedPaintEllipse.create_shape(
    center=(250, 450), radius_x=200, radius_y=400, name='ellipse')
circle = CeedPaintCircle.create_shape(
    center=(700, 300), radius=200, name='circle')
polygon = CeedPaintPolygon.create_shape(
    points=[275, 300, 700, 300, 500, 800], name='polygon')

# add the shapes to factory
shape_factory.add_shape(ellipse)
shape_factory.add_shape(circle)
shape_factory.add_shape(polygon)

# create group, name it, and add shapes to it
group = shape_factory.add_group()
group.name = 'shapes'
group.add_shape(ellipse)
group.add_shape(polygon)

# create a function. We get it from the factory because that's where Ceed gets
# it from, but we use the type hint to help the IDE type it correctly (not
# required by python though)
func_cls: Type[ConstFunc] = function_factory.get('ConstFunc')
func = func_cls(
    function_factory=function_factory, name='stable', duration=1, a=0.5)

# create a stage. We get it from the factory for the same reason as the function
stage_cls: Type[CeedStage] = stage_factory.get('CeedStage')
stage = stage_cls(
    function_factory=function_factory, stage_factory=stage_factory,
    shape_factory=shape_factory, name='best stage')

# now add the shape, group, and function to the stage
stage.add_shape(circle)
stage.add_shape(group)
stage.add_func(func)

# we do need to add the stage to the factory, but we don't need to do it for
# the function because we don't need it in the global function list
stage_factory.add_stage(stage)

# save it to disk for import later
CeedDataWriterBase.save_config_to_yaml(
    'config.yml', shape_factory=shape_factory,
    function_factory=function_factory, stage_factory=stage_factory)
