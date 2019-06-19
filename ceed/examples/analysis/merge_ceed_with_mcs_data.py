"""
Merging Ceed Data with MCS Data
================================

The following script shows how to merge the ceed h5 data file that contains all
the experimental data from the projector stimulation side, with the multi-array
electrode (MEA) data recorded by the MCS system.

The MCS data must be first exported from the native MCS format to the h5
format.
"""
from ceed.analysis.merge_data import CeedMCSDataMerger
import os.path

# The ceed file with the projector experimental data
ceed_file = '../data/experiment_data_ceed.h5'
# The h5 file exported by MCS containing the MEA data
mcs_file = '../data/experiment_data_mcs.h5'
if not os.path.exists(ceed_file) or not os.path.exists(mcs_file):
    raise Exception(
        'Data not found, please manually extract ceed/examples/data/experiment'
        '_data.zip to ceed/examples/data/')

# The ceed file that will be generated containing the time aligned and merged
# ceed data and MEA data between the two systems
output_file = '../data/experiment_data_merged.h5'
# Optional text notes that will be integrated into the merged file
notes = """
At time x, the slice woke up
"""
# Optional full filename of a text file containing notes that will be
# integrated into the merged file. This will be appended to the notes above
notes_filename = None

# how to align the MCS to the ceed data. It could be none, in which case we'll
# use the pattern that ceed sends on the digital channel to the MCS recording
# system. If it's a number, we'll try to align it using the timestamps of the
# ceed and MCS data. The number is then the estimated delay in seconds in the
# MCS file where the ceed stage started playing
find_by = None
# class that actually merges the two files
merger = CeedMCSDataMerger()

# read the MCS data - MCS data is one long file containing the data for all the
# ceed stages that played. Ceed however, splits the data into stages
merger.read_mcs_digital_data(mcs_file)
# we only need to parse the mcs data once, after the ceed data has been parsed.
# If we provide a direct alignment time offset, it needs to be re-parsed for
# every experiment in the ceed file
init = False

# this dictionary will accumulate the alignment metadata for all the
# experiments in the data files. Each item is an experiment
alignment = {}
# `get_experiment_numbers` lists all the experiments in the ceed file
# bad experiments to be ignored can be added to the `ignore_list`
for experiment in merger.get_experiment_numbers(ceed_file, ignore_list=[]):
    # read and parse the ceed data for this experiment
    merger.read_ceed_digital_data(ceed_file, experiment)
    merger.parse_ceed_digital_data()

    if not init or find_by is not None:
        # parse the mcs data - it has to happen after ceed data is parsed
        merger.parse_mcs_digital_data(find_by=find_by)
        init = True

    try:
        # compute the alignment
        align = alignment[experiment] = merger.get_alignment(
            find_by=find_by)
        print(
            'Aligned MCS and ceed data for experiment {} at MCS samples '
            '[{} - {}] ({} frames)'.format(
                experiment, align[0], align[-1], len(align)))
    except Exception as e:
        print(
            "Couldn't align MCS and ceed data for experiment "
            "{} ({})".format(experiment, e))

# now actually merge the files and include the alignment metadata
merger.merge_data(
    output_file, ceed_file, mcs_file, alignment, notes=notes,
    notes_filename=notes_filename)
