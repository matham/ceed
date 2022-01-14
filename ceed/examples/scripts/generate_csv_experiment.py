"""This script shows how to generate a yaml config containing a CSV stage and
a normal stage with a CSV function. It also shows how to generate the CSV
data that can be used by the stage and function.

It generates three files:
- csv_experiment_config.yml containing the yaml config to be imported
- experiment_stage.csv containing the csv data that needs to be selected in the
  ``csv_stage`` stage's settings in the GUI
- experiment_function.csv containing the csv data that needs to be selected in
  the csv_func_stage stage's function settings in the GUI

Either the csv_stage or the csv_func_stage can then be run and the values
will be read from the appropriate csv file.

See the docs for the CSVStage and CSVFunc for details on the format of the CSV
files.
"""
from ceed.shape import CeedPaintCanvasBehavior, CeedPaintCircle, \
    CeedPaintEllipse
from ceed.function import FunctionFactoryBase, register_all_functions
from ceed.stage import StageFactoryBase, register_all_stages
from ceed.storage.controller import CeedDataWriterBase
import csv
import math

# create shape/function/stage factory
shape_factory = CeedPaintCanvasBehavior()
function_factory = FunctionFactoryBase()
stage_factory = StageFactoryBase(
    function_factory=function_factory, shape_factory=shape_factory)

# now we need to register all the builtin functions and stage plugins
register_all_functions(function_factory)
register_all_stages(stage_factory)

# create the shapes and name them
ellipse = CeedPaintEllipse.create_shape(
    center=(250, 450), radius_x=200, radius_y=400, name='ellipse')
circle = CeedPaintCircle.create_shape(
    center=(700, 300), radius=200, name='circle')

# add the shapes to factory
shape_factory.add_shape(ellipse)
shape_factory.add_shape(circle)


def create_csv_stage():
    # create a csv stage that will have two shapes and accept a csv file
    # that is generated below
    csv_stage = stage_factory.get('CSVStage')(
        function_factory=function_factory, stage_factory=stage_factory,
        shape_factory=shape_factory, name='csv_stage', color_r=True,
        color_g=True, color_b=True)
    stage_factory.add_stage(csv_stage)

    # generate the csv file for the stage, ellipse is going to be red, circle
    # yellow
    with open('experiment_stage.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['ellipse_red', 'circle_red', 'circle_green'])
        for i in range(1000):
            val = .5 * math.sin(i / 120 * 2 * math.pi) + .5
            writer.writerow([val, val, val])


def create_csv_function_stage():
    # create a stage that will have two shapes and one csv function. The csv
    # function will accept a csv file that is generated below
    func = function_factory.get('CSVFunc')(
        function_factory=function_factory, name='csv')

    stage = stage_factory.get('CeedStage')(
        function_factory=function_factory, stage_factory=stage_factory,
        shape_factory=shape_factory, name='csv_func_stage', color_b=True)
    stage_factory.add_stage(stage)
    stage.add_func(func)
    stage.add_shape(circle)
    stage.add_shape(ellipse)

    # generate the csv file for the function; ellipse and circle are going to
    # be red
    with open('experiment_function.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        for i in range(1000):
            writer.writerow([.5 * math.sin(i / 120 * 2 * math.pi) + .5])


# create the stages/functions and csv files
create_csv_stage()
create_csv_function_stage()

# save it to disk for import later
CeedDataWriterBase.save_config_to_yaml(
    'csv_experiment_config.yml', shape_factory=shape_factory,
    function_factory=function_factory, stage_factory=stage_factory,
    overwrite=True)
