"""
Plotting Ceed Experiment Data
=============================

This script shows how to read the ceed data stored in an h5 file and then
plot it showing the experiment that was played as well as the MEA data.
"""
from ceed.analysis import CeedDataReader
import matplotlib.pyplot as plt
import numpy as np

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

print('Making plots')
f, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

print('Showing electrode data')
electrode = sorted(reader.electrodes_data.keys())[0]
metadata = reader.electrodes_metadata[electrode]
offset, scale = reader.get_electrode_offset_scale(electrode)
freq = metadata['sampling_frequency']
unit = metadata['Unit']
data = (np.array(reader.electrodes_data[electrode]) - offset) * scale

ax2.plot(np.arange(len(data)) / freq, data, label=electrode)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Electrode amplitude ({})'.format(unit))

colors = {'r': 0, 'g': 1, 'b': 2, 'a': 3}
channel = 'b'
print('Showing ceed experiment data LED channel "{}"'.format(channel))
for exp in reader.experiments_in_file:
    # load a particular experiment
    reader.load_experiment(exp)
    print('Loaded experiment {}'.format(reader.loaded_experiment))
    print('Stage run for this experiment is "{}"'.format(
        reader.experiment_stage_name))
    if reader.electrode_intensity_alignment is None:
        print('Ceed-MCS alignment not found in the file. Did you forget to '
              'merge it into the ceed data file?')

    shape_t = np.array(reader.electrode_intensity_alignment) / freq
    for shape, intensity in reader.shapes_intensity.items():
        ax1.plot(shape_t, np.array(intensity)[:len(shape_t), colors[channel]],
                 label='{}-{}'.format(shape, reader.experiment_stage_name))

ax1.set_ylabel('Amplitude')
ax1.legend()
plt.show()

print('Closing data file')
reader.close_h5()
