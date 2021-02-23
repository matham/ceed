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

# class that actually merges the two files
merger = CeedMCSDataMerger(ceed_filename=ceed_file, mcs_filename=mcs_file)

# read the MCS data - MCS data is one long file containing the data for all the
# ceed experiments that played. Ceed however, splits the data into experiments
merger.read_mcs_data()
# read the overall data from this file, including the Ceed-MCS data link
# configuration required to be able to align the two files
merger.read_ceed_data()
# this parses the Ceed-MCS data-link data adn tries to break the MCS data into
# experiments so we can later locate individual experiments for alignment
merger.parse_mcs_data()

# this dictionary will accumulate the alignment metadata for all the
# experiments in the data files. Each item is an experiment
alignment = {}
# `get_experiment_numbers` lists all the experiments in the ceed file
# bad experiments to be ignored can be added to the `ignore_list`
for experiment in merger.get_experiment_numbers(ignore_list=[]):
    # read and parse the ceed data for this experiment
    merger.read_ceed_experiment_data(experiment)
    merger.parse_ceed_experiment_data()

    try:
        # compute the alignment
        align = alignment[experiment] = merger.get_alignment()
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
    output_file, alignment, notes=notes, notes_filename=notes_filename)
