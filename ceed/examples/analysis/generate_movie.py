"""
Generating Movie from Ceed Experiment Data
==========================================

This script shows how to read the ceed data stored in an h5 file and then
generate a movie showing the experiment that was played as well as the MEA
data.
"""
from ceed.analysis import CeedDataReader

ceed_data = '../data/experiment_data_merged.h5'

# create instance that can load the data
reader = CeedDataReader(ceed_data)
print('Created reader for file {}'.format(reader.filename))
# open the data file
reader.open_h5()

# get the number of experiments and images stored in the file. These images
# are in addition to the camera images stored with each experiment
print('Experiment names found in the file: {}'.format(
      reader.experiments_in_file))

# load the mcs data into memory
reader.load_mcs_data()
if not reader.electrodes_data:
    print('Electrode data not found in the file. Did you forget to merge it '
          'into the ceed data?')
if reader.electrode_dig_data is None:
    print('Electrodes digital data not found in the file')


# load a particular experiment
reader.load_experiment(1)
print('Loaded experiment {}'.format(reader.loaded_experiment))
print('Stage run for this experiment is {}'.format(
    reader.experiment_stage_name))
if reader.electrode_intensity_alignment is None:
    print('Ceed-MCS alignment not found in the file. Did you forget to merge '
          'it into the ceed data file?')

# experiments normally have a camera image of the slice florescence before the
# experiment - you can save it to a file. Only bmp is supported (I think)
print('Saving background camera image at '
      '../data/experiment_slice_florescence.bmp')
reader.save_image(
    '../data/experiment_slice_florescence.bmp', reader.experiment_cam_image)

# this list is a list of functions that is past to the movie generator
# function, each of the functions are called for every frame and are given the
# opportunity to show something
paint_funcs = [
    # this function displays the experiment's background image in the
    # background at the appropriate orientation as in the experiment
    reader.paint_background_image(
        reader.experiment_cam_image,
        transform_matrix=reader.view_controller.cam_transform),
    # this function displays the experiment's MEA electrode center locations
    # at the appropriate orientation as in the experiment
    reader.show_mea_outline(reader.view_controller.mea_transform),
    # this function shows the electrode voltage data
    reader.paint_electrodes_data_callbacks(
        reader.get_electrode_names(), draw_pos_hint=(1, 0), volt_axis=50)
]

print('Generating movie at ../data/experiment_slice_data movie.mp4, please '
      'be patient')
reader.generate_movie(
    '../data/experiment_slice_data movie.mp4',
    lum=1,
    # make the video twice as large in the x direction as the projector so we
    # can show the voltage data to its right
    canvas_size_hint=(2, 1),
    # show the data at the normal speed
    speed=1.,
    paint_funcs=paint_funcs
)

print('Done generating movie, closing data file')
reader.close_h5()
