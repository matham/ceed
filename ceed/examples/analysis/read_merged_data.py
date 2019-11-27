"""
Reading Ceed Experiment Data
============================

This script shows how to read the ceed data stored in an h5 file, after it has
been mrged with the MCS multi-electrode array (MEA) data.
"""
from ceed.analysis import CeedDataReader
from pprint import pprint

ceed_data = '../data/experiment_data_merged.h5'

# create instance that can load the data
reader = CeedDataReader(ceed_data)
print('Created reader for file {}'.format(reader.filename))
# open the data file
reader.open_h5()

# once loaded, the version and the overall file log and file notes is available
print('-' * 20)
print('The ceed version of the file is {}'.format(reader.ceed_version))
print('The file log is:\n{}\n'.format(reader.app_logs.strip()))
print('The file notes is:\n{}\n'.format(reader.app_notes.strip()))

# get the number of experiments and images stored in the file. These images
# are in addition to the camera images stored with each experiment
print('Experiment names found in the file: {}'.format(
      reader.experiments_in_file))

print('Number of saved images found in the file (excluding images within the '
      'experiments): {}'.format(reader.num_images_in_file))
for i in range(reader.num_images_in_file):
    img, notes, save_time = reader.get_image_from_file(i)
    print('Image {}, create at time={}, notes:\n{}\n'.format(
        img, save_time, notes))
print('-' * 20)


# load the mcs data into memory
reader.load_mcs_data()
electrodes = sorted(reader.electrodes_data.keys())
if electrodes:
    e0 = electrodes[0]
    print('Electrodes found in the file: {}'.format(electrodes))
    print('Electrode "{}" data array shape is {}'.format(
        e0, reader.electrodes_data[e0].shape))

    s0, s1 = reader.electrodes_data[e0][0], reader.electrodes_data[e0][-1]
    print('Electrode "{}" first and last raw data samples are {}, {}'.format(
        e0, s0, s1))
    offset, scale = reader.get_electrode_offset_scale(e0)
    unit = reader.electrodes_metadata[e0]['Unit']
    print('Electrode "{0}" first and last data samples are {1}{3}, {2}{3}'.
          format(e0, (s0 - offset) * scale, (s1 - offset) * scale, unit))

    print('Electrode "{}" metadata:'.format(e0))
    pprint(reader.electrodes_metadata[e0])
else:
    print('Electrode data not found in the file. Did you forget to merge it '
          'into the ceed data file?')

if reader.electrode_dig_data is not None:
    print('Electrodes digital data shape is {}'.format(
        reader.electrode_dig_data.shape))
else:
    print('Electrodes digital data not found in the file')
print('-' * 20)


# load a particular experiment
reader.load_experiment(1)
print('Loaded experiment {}'.format(reader.loaded_experiment))
print('Stage run for this experiment is "{}"'.format(
    reader.experiment_stage_name))

if reader.electrode_intensity_alignment is None:
    print('Ceed-MCS alignment not found in the file. Did you forget to merge '
          'it into the ceed data file?')
else:
    print('Ceed-MCS alignment indices shape is {}'.format(
        reader.electrode_intensity_alignment.shape))

# print the experiments notes, if any
print('The experiment notes is:\n{}\n'.format(reader.experiment_notes.strip()))
print('Experiment start is at time={}'.format(reader.experiment_start_time))
print('Experiment camera image={}'.format(reader.experiment_cam_image))

shapes = sorted(reader.shapes_intensity.keys())
print('Found the following shapes in the experiment: {}'.format(shapes))
print('The intensity array for shape "{}" is size {}'.format(
    shapes[0], reader.shapes_intensity[shapes[0]].shape))

print('The array recording the projector LED state is shape {}'.format(
    reader.led_state.shape))

print('Closing data file')
reader.close_h5()
