"""Ceed-MCS data merging
=========================

During an experiment both Ceed and MCS record their respective data to separate
files. Ceed records the projected frames and all its shapes to an H5 file, while
MCS records its electrode and digital data to a proprietary file format that can
be exported to H5.

This module merges and temporally aligns the Ceed frames to the MCS electrode
data and creates a single H5 file containing the data from both files.
This new file can be read with :class:`~ceed.analysis.CeedDataReader`

.. _handshake-protocol:

Temporal alignment protocol
---------------------------

As explained in :class:`~ceed.storage.controller.DataSerializerBase`,
during a Ceed experiment Ceed generates upto 24-bits per projector frame that is
recorded into the MCS data stream. Following is a basic summary as it relates to
merging.

Each frame contains upto 24-bits of data labeling the frame. There's a clock
that alternates high/low for each frame. There's a short counter of a few bits
that increments by one for each displayed frame (not including dropped frames)
and overflows back to zero. This allows us to detect bad frames/data. And
finally there's a large n-bit counter int broken up into groups of a few bits,
and each frame contains the next group of these bits until the full int was sent
so it can start on the next counter value.

E.g. if it's 32-bit and we have 2 bits per frame, then naively it takes 16
frames to send the full int, broken up into groups of two. However, each set of
bits is sent twice, the bits followed by their one's complement. So it takes
32 frames to send the full int as follows: b0-1, b0-1_comp, b2-3, b2-3_comp,
..., b30-31, b30-31_comp. This allows us to further detect data corruption.

So, in an experiment as we send the global counter, starting from one it may
look as follows. At frame zero we start sending the counter of 1. This takes 32
frames to send. So, at frame 32 when we're done with the int we start sending
the next current value of the counter - 33. This is sent over the next 32 frames
etc.

Handshake
^^^^^^^^^

There's one further caveat. At the start of the experiment Ceed sends some
handshaking data before starting to send the counter ints. Specifically, the
first int value is the number of ints in the subsequent handshake. The
subsequent handshake is a sequence of ints, composed of bytes that are a
unique signature of that experiment. This is how we can later locate a Ceed
experiment in the MCS data that contains multiple experiments.

Once the handshake is complete, Ceed starts sending the counter values of the
frames as explained above. However, unlike the counter ints that are sent
along with their complement, for the initial length int and handshake data the
groups are sent twice. I.e. b0-1, b0-1, b2-3, b2-3, ..., b30-31, b30-31. This
allows us to locate the start of an experiment in the MCS data.

.. _serializer-sub-frames:

Sub-frames
^^^^^^^^^^

When Ceed is running in quad mode where each frame has sub-frames (e.g. quad4x
each frame is actually 4 frames), the digital IO is only updated for each major
frame and it's the same for all the frame's sub-frames. So although Ceed records
data for each sub frame, the serializer digital IO is only updated for each
major frame. So the counter increments with the number of sub-frames times the
number of frames it took to transmit the int - because that's how may frames
completed while sending a int.

However, MCS can only see when a major frame changes because that's when the
digital IO changes. So we have to infer and break each main frame into
sub-frames to find where in the electrode data the sub-frames were rendered.

Dropped frames
^^^^^^^^^^^^^^

As explained in :ref:`dropped-frames`, Ceed will sometimes drop a frame. The
Ceed frame counter (denoting the time as count / frame rate) still increments
for the frame, but the frame is not displayed. Dropping a frame has no effect
on the digital IO and it is continuous like normal (we don't update the IO for
the dropped frame). The only visible difference is that the counter int will
have incremented by a larger values than normal when we send the next counter
int after finishing the current int. That's is how MCS can detect a dropped
frame, in addition to observing that a Ceed frame went too long because the IO
is not updated at the normal rate, like the
:class:`~ceed.view.controller.TeensyFrameEstimation` detection approach.

.. _merging-ceed-mcs:

Parsing Ceed data
-----------------

Ceed logs all its data into the H5 file. When parsing an experiment using
:class:`CeedDigitalData`, it takes the digital IO data logged by Ceed, including
the data for all the sub-frames and dropped frames, and it extracts the frame
clock, short and long counter values for each frame. This gives us the expected
value for each of these three signals for every frame.

.. _parse-ceed-mcs:

Parsing MCS data
----------------

Unlike parsing the Ceed data, the MCS data contains data for multiple
experiments so when parsing we first have to split them into experiments and
then we parse each experiment. Furthermore, each Ceed main frame is
approximately 1 / 120 seconds long. MCS generally samples at multiple kHz.
So MCS will typically read many duplicate values for each Ceed frame. This
needs to be de-duplicated before parsing.

Once the MCS has been de-duplicated and split into experiments, we can parse
it like we parse the Ceed data and get the corresponding counter value for
each detected frame.

When parsing the MCS data we assume that there's a pause of at least 10 frames
between experiments, during which the clock is constant. This simplifies
locating the start of experiments. Additionally, we cannot parse experiments
which are shorter than a couple of frames.

Merging
-------

When we have a Ceed experiment parsed both for the Ceed and MCS data, we can
then relate and find exactly where in the electrode data a Ceed frame is
displayed. This is returned by :meth:`CeedMCSDataMerger.get_alignment` and
is stored in the new file.

Typically we'd search in the parsed MCS data for the unique experiment's
handshake value as provided by Ceed's data. However, if an experiment was
stopped early and is too short to have sent the full handshake. So multiple
experiments could potentially share the same handshake, if the initial bytes
were the same. In that case the user would need to disambiguate between them
somehow.

Merging example
---------------

The typical steps for merging is to instantiate a :class:`CeedMCSDataMerger`
with the Ceed and MCS H5 files. Then :meth:`CeedMCSDataMerger.read_mcs_data`,
:meth:`CeedMCSDataMerger.read_ceed_data`, and
:meth:`CeedMCSDataMerger.parse_mcs_data`.

Typically a Ceed and MCS file will contain multiple experiments.
Using :meth:`CeedMCSDataMerger.get_experiment_numbers` you'd get all the
experiments in the Ceed file, ignoring any experiment which are not in the
MCS file, or bad experiments. For each of the experiments we'd
:meth:`CeedMCSDataMerger.read_ceed_experiment_data` and
:meth:`CeedMCSDataMerger.parse_ceed_experiment_data` for that experiment.

Once parsed we can get the alignment with
:meth:`CeedMCSDataMerger.get_alignment` and save it in a dict. Finally,
once we have the alignment for all the experiments we can create the output
file with all the data using :meth:`CeedMCSDataMerger.merge_data`.

Following is a worked example::

    ceed_file = 'ceed_data.h5'
    mcs_file = 'mcs_data.h5'
    output_file = 'ceed_mcs_merged.h5'
    notes = ''
    notes_filename = None  # no notes

    # create merge object
    merger = CeedMCSDataMerger(ceed_filename=ceed_file, mcs_filename=mcs_file)

    # read ceed and MCS data and parse MCS metadata
    merger.read_mcs_data()
    merger.read_ceed_data()
    merger.parse_mcs_data()

    alignment = {}
    # get alignment for all interesting experiments
    for experiment in merger.get_experiment_numbers(ignore_list=[]):
        # read and parse the data
        merger.read_ceed_experiment_data(experiment)
        merger.parse_ceed_experiment_data()

        # try to get the Ceed-MCS alignment
        try:
            align = alignment[experiment] = merger.get_alignment()
            # print experiment summary, see method for column meaning
            print(merger.get_skipped_frames_summary(align, experiment))
        except Exception as e:
            print(
                "Couldn't align MCS and ceed data for experiment "
                "{} ({})".format(experiment, e))

    # finally create new file from alignment and existing files
    merger.merge_data(
        output_file, alignment, notes=notes, notes_filename=notes_filename)
"""

import sys
import datetime
from tqdm import tqdm
from collections import defaultdict
import os.path
from math import ceil
from typing import List, Dict, Optional, Tuple, Union
from shutil import copy2
from McsPy import ureg
import McsPy.McsData
import numpy as np
import nixio as nix
from more_kivy_app.utils import yaml_dumps, yaml_loads

__all__ = (
    'CeedMCSDataMerger', 'AlignmentException', 'BitMapping32',
    'DigitalDataStore', 'MCSDigitalData', 'CeedDigitalData')

McsPy.McsData.VERBOSE = False


class AlignmentException(Exception):
    """Raised if the Ceed-MCS data cannot be aligned.
    """
    pass


class BitMapping32:
    """Maps 32-bit integers to other 32-bit integers, assuming each bit of the
    input is in the output, but perhaps at a different index.

    E.g. if we have a 32-bit int whose bits at indices 5 and 8 are exchanged,
    or if bit 5 should be moved to bit 0, and all other bits ignored, this can
    achieve it.

    As input it takes a list of indices that are less than 32, the bit index
    represented by each item in the list is mapped to the bit-index represented
    by the index of the item in the input list.

    E.g. to shift all bits upto 10 to the right by two bits (chopping off the
    first two bits which don't get mapped)::

        >>> mapping = BitMapping32([i + 2 for i in range(10)])
        >>> data = np.array([0, 1, 2, 4, 8, 16], dtype=np.int32)
        >>> mapping.map(data)
        array([0, 0, 0, 1, 2, 4], dtype=uint32)

    This mapped bits 0-1 to nothing (they are not on the list), bit 2 to 0,
    3 to 1 etc.

    .. note::

        This computes the mapping as 4 16-bit int arrays, each containing
        ``2 ^ 16`` items (stored internally). This could be aproblem on low
        memory devices.
    """

    l_to_l: np.ndarray = None
    """Maps the lower 16-bits of the input to the lower 16-bits of the output.
    """

    h_to_l: np.ndarray = None
    """Maps the upper 16-bits of the input to the lower 16-bits of the output.
    """

    l_to_h: np.ndarray = None
    """Maps the lower 16-bits of the input to the upper 16-bits of the output.
    """

    h_to_h: np.ndarray = None
    """Maps the upper 16-bits of the input to the upper 16-bits of the output.
    """

    def __init__(self, bits: List[int]):
        self._compute_maps(bits)

    def _compute_maps(self, bits: List[int]):
        """Computes the mappings.
        """
        indices = np.arange(2 ** 16, dtype=np.uint16)

        l_to_l = np.zeros((2 ** 16, ), dtype=np.uint16)
        h_to_l = np.zeros((2 ** 16, ), dtype=np.uint16)
        l_to_h = np.zeros((2 ** 16, ), dtype=np.uint16)
        h_to_h = np.zeros((2 ** 16, ), dtype=np.uint16)

        for i, k in enumerate(bits):
            if i <= 15:
                if k <= 15:
                    l_to_l |= ((indices & (1 << k)) >> k) << i
                else:
                    h_to_l |= \
                        (indices & (1 << (k % 16))) >> (k % 16) << i
            else:
                if k <= 15:
                    l_to_h |= ((indices & (1 << k)) >> k) << i
                else:
                    h_to_h |= \
                        (indices & (1 << (k % 16))) >> (k % 16) << i

        self.l_to_l = l_to_l
        self.h_to_l = h_to_l
        self.l_to_h = l_to_h
        self.h_to_h = h_to_h

    def map(self, data: np.ndarray) -> np.ndarray:
        """Maps the 32-bit integer input data to the 32-bit integer output data
        according to the given mapping. See :class:`BitMapping32`.
        """
        mapped = np.zeros(len(data), dtype=np.uint32)
        # get upper bits and shift up
        mapped |= self.l_to_h[data & 0xFFFF] | \
            self.h_to_h[(data >> 16) & 0xFFFF]
        mapped <<= 16
        # get lower bits
        mapped |= self.l_to_l[data & 0xFFFF] | \
            self.h_to_l[(data >> 16) & 0xFFFF]

        return mapped


class DigitalDataStore:
    """The class that can parse the MCS and Ceed upto 24-bit data pattern
    associated with each Ceed frame.

    It takes the recorded data and parses and decomposes it into the clock bit,
    the :attr:`~ceed.storage.controller.DataSerializerBase.short_count_indices`
    bits, and the
    :attr:`~ceed.storage.controller.DataSerializerBase.count_indices`.

    This lets us then compare the decomposed data between the Ceed data output
    through the Propixx system and recorded by Ceed for each frame and the MCS
    data recorded through the Propixx-MCS data link. Then we can align each
    frame to its location in the MCS data by comparing them (especially since
    Ceed sends frame numbers over the counter bits).
    """

    counter_bit_width: int = 32
    """The number of bits in the counter sent over the long counter.
    See :attr:`~ceed.storage.controller.DataSerializerBase.counter_bit_width`.
    """

    short_map: BitMapping32 = None
    """The mapping that maps the
    :attr:`~ceed.storage.controller.DataSerializerBase.short_count_indices`
    of the recorded data to a contiguous bit pattern representing the number.

    I.e. the recorded data could store the n-bit counter (e.g. 4-bits) over
    some random bit arrangement (e.g. bits 5, 7, 2, 9). This will map it to
    the lower n-bits in the correct order (e.g. bits 0-3) so the value can just
    be read off.
    """

    short_count_indices: List[int] = None
    """See
    :attr:`~ceed.storage.controller.DataSerializerBase.short_count_indices`.
    """

    short_count_data: np.ndarray = None
    """The short counter data parsed and mapped so it's represented
    as a correct number that can just be read for its value.
    """

    count_map: BitMapping32 = None
    """Same as :attr:`short_map`, but for
    :attr:`~ceed.storage.controller.DataSerializerBase.count_indices`.
    """

    count_indices: List[int] = None
    """See
    :attr:`~ceed.storage.controller.DataSerializerBase.count_indices`.
    """

    count_data: np.ndarray = None
    """The long counter data parsed and mapped so it's represented
    as a correct number that can just be read for its value.

    Keep in mind, that although it has been mapped to the lower n-bits,
    the long counter is a :attr:`counter_bit_width` integer split over multiple
    frames so this value is not quite a complete number.
    """

    clock_index: int = None
    """See
    :attr:`~ceed.storage.controller.DataSerializerBase.clock_idx`.
    """

    clock_data: np.ndarray = None
    """The clock data parsed and mapped so it is represented as either zero or
    one, depending on the state of the clock.
    """

    data: np.ndarray = None
    '''The raw input data that contains the clock, short, and long counter data
    across its various bits.

    When frames are dropped by Ceed, the
    :class:`~ceed.storage.controller.DataSerializerBase` will be correctly
    incremented. This means the long counter will skip those frames. But the
    data will not be corrupt because we split the long counter over multiple
    frames and we still correctly send each count even if frames are globally
    skipped. The only difference is that when the next count value is split
    over the frames, that value will have incremented more than normal.

    The short counter's behavior doesn't change when frames are dropped
    because it increments for each frame displayed.
    '''

    expected_handshake_len: int = 0
    """The number of bytes required to send the full handshake as described
    in :class:`~ceed.storage.controller.DataSerializerBase`.

    The handshake always begins with the number of ints of size
    :attr:`counter_bit_width` that will make up the handshake (see
    :ref:`handshake-protocol`). This stores the
    number of bytes and is computed from that (assuming we were able to parse
    the first handshake int).

    It is automatically set.
    """

    handshake_data: bytes = b''
    """The parsed handshake (config_bytes) data sent at the start of each
    experiment as described in
    :meth:`~ceed.storage.controller.DataSerializerBase.get_bits`.

    It is automatically parsed from the data.
    """

    counter: np.ndarray = None
    """The long counter value associated with each frame. This is provided in
    the Ceed data, since Ceed knows this as it records it for each frame.
    """

    def __init__(self, short_count_indices, count_indices, clock_index,
                 counter_bit_width):
        """indices are first mapped with projector_to_aquisition_map. """
        super(DigitalDataStore, self).__init__()
        self.short_count_indices = short_count_indices
        self.count_indices = count_indices
        self.clock_index = clock_index
        self.counter_bit_width = counter_bit_width

        self.short_map = BitMapping32(short_count_indices)
        self.count_map = BitMapping32(count_indices)

    def _parse_components(self, data: np.ndarray) -> None:
        self.data = data
        clock_index = self.clock_index
        clock_bit = 1 << clock_index

        self.clock_data = (data & clock_bit) >> clock_index
        self.short_count_data = self.short_map.map(data)
        self.count_data = self.count_map.map(data)

    def compare(
            self, short_count_indices, count_indices, clock_index):
        """Compares the similarly named properties of this instance to
        thee given values and returns if they are the same.

        If it is the same, we assume we can just re-use this instance and
        its mappings (which are expensive to compute).
        """
        if self.short_count_indices is None and \
                short_count_indices is not None:
            return False
        if self.short_count_indices is not None \
                and short_count_indices is None:
            return False
        if self.short_count_indices is not None:
            if not np.array_equal(
                    self.short_count_indices, short_count_indices):
                return False

        if self.count_indices is None and count_indices is not None:
            return False
        if self.count_indices is not None and count_indices is None:
            return False
        if self.count_indices is not None:
            if not np.array_equal(self.count_indices, count_indices):
                return False

        if self.clock_index != clock_index:
            return False
        return True

    def __eq__(self, other):
        if not isinstance(other, DigitalDataStore):
            return False

        return self.compare(
            other.short_count_indices, other.count_indices, other.clock_index)

    @property
    def n_parts_per_int(self):
        """The number of parts into which a complete int of the long
        counter (of size :attr:`counter_bit_width`) will have to be split into
        so we can send it over the bits available for the long counter.
        """
        return int(
            ceil(self.counter_bit_width / len(self.count_indices)))

    @property
    def n_bytes_per_int(self):
        """The number of bytes (rounded down) required to represent the
        long counter :attr:`counter_bit_width` bits.
        """
        return self.counter_bit_width // 8

    def get_count_ints(
            self, count_data: np.ndarray, contains_sub_frames: bool,
            n_sub_frames: int
    ):
        """Takes raw :attr:`count_data`, bool indicating whether the count data
        contains sub-frames data as well as the number of sub-frames and
        it parses it into complete counter data.

        In quad mode, Ceed can generate 4 or 12 frames for each GPU frame.
        However, the count data may contain data for each whole frame only, or
        for all frames, including sub-frames. ``contains_sub_frames`` is True
        for the latter and ``n_sub_frames`` indicates the number of sub-frames
        per frame (1 if there are no sub-frames). See
        :ref:`serializer-sub-frames`.

        It returns a three tuple, ``(count, count_2d, count_inverted_2d)``.
        As explained in :ref:`handshake-protocol, each long counter int is sent
        broken across multiple frames, in addition to being sent inverted
        (except for handshake that is not inverted).
        So each int is sent in parts as k1, k1_inv, k2, k2_inv, ..., kn, kn_inv.

        Defining ``M = len(count_data) // n // 2`` (the number of whole ints).
        ``count`` is the collated up-to 64-bit int that contains all the 1..n
        components and is size M. ``count_2d`` is a 2D array of Mxn containing
        all the parts. ``count_inverted_2d`` similarly of size Mxn contains only
        the inverted parts.
        """
        # last value in count_data could be spurious (from mcs)
        n_counter_bits = len(self.count_indices)
        n_parts_per_int = self.n_parts_per_int
        # all sub frames are the same (if we have them)
        # We don't need inverted frames
        count = count_data
        if contains_sub_frames:
            count = count_data[::n_sub_frames]

        count_inverted_2d = count[1::2]
        count = count[::2]

        # only keep full ints
        remainder = len(count) % n_parts_per_int
        if remainder:
            count = count[:-remainder]
        remainder = len(count_inverted_2d) % n_parts_per_int
        if remainder:
            count_inverted_2d = count_inverted_2d[:-remainder]

        # it comes in little endian as ints of counter_bit_width size
        count = np.reshape(count, (-1, n_parts_per_int))
        count_inverted_2d = np.reshape(count_inverted_2d, (-1, n_parts_per_int))
        # should have same height
        n = min(count.shape[0], count_inverted_2d.shape[0])
        count = count[:n, :]
        count_inverted_2d = count_inverted_2d[:n, :]
        assert count.shape == count_inverted_2d.shape

        # if inverted has one row less than count, chop it off
        count_2d = count.copy()
        if count_2d.shape[0] != count_inverted_2d.shape[0]:
            count_2d = count_2d[:-1, :]

        # we use 64 bit as max value for counter size
        count = count.astype(np.uint64)
        for i in range(n_parts_per_int):
            count[:, i] <<= i * n_counter_bits
        count = np.bitwise_or.reduce(count, axis=1)

        return count, count_2d, count_inverted_2d

    def check_counter_consistency(
            self, count_data, count_2d, count_inverted_2d, n_handshake_ints,
            contains_sub_frames, n_sub_frames, exclude_last_value):
        """Takes the parsed data and verifies that the data is consistent and
        doesn't contain errors as best as it can.
        """
        if count_2d.shape[0]:
            # even if no handshake sent, the first value will be size (zero)
            # so we always have at least one item
            assert n_handshake_ints
        count_shape = count_2d.shape

        if contains_sub_frames:
            count_data = count_data[::n_sub_frames]
        # remove data already in count_2d
        count_data = count_data[count_shape[0] * count_shape[1] * 2:]
        count_end = has_trailing = len(count_data)
        # we can't check the last item if it doesn't have inverted value
        if count_end % 2:
            count_end -= 1
        elif exclude_last_value:
            count_end -= 2
            count_end = max(count_end, 0)

        # set count bits
        count_bits = 0
        for i in range(len(self.count_indices)):
            count_bits |= 1 << i

        # do we have data beyond the handshake
        has_counter = count_shape[0] > n_handshake_ints
        # if exclude final, see if data extends past handshake
        handshake_end = n_handshake_ints
        if not has_trailing and not has_counter and exclude_last_value:
            handshake_end -= 1

        # if exclude final, see if data extends past counter
        count_2d_end = count_shape[0]
        if not has_trailing and exclude_last_value:
            count_2d_end -= 1

        # config is not inverted
        if not np.all(count_2d[:handshake_end, :]
                      == count_inverted_2d[:handshake_end, :]):
            raise AlignmentException('Handshake data corrupted')

        # first item is not inverted
        if not np.all(count_2d[:count_2d_end, 0]
                      == count_inverted_2d[:count_2d_end, 0]):
            raise AlignmentException('Non-inverted data group corrupted')

        # remaining counter is inverted
        if not np.all(
                (count_2d[n_handshake_ints:count_2d_end, 1:] ^ count_bits)
                & count_bits
                == count_inverted_2d[n_handshake_ints:count_2d_end, 1:]):
            raise AlignmentException('Inverted data group corrupted')

        # n_handshake_ints >= 1, if we have at least one full int
        if not n_handshake_ints:
            # only have partial int, handshake is not inverted
            assert not count_shape[0]
            if not np.all(
                    count_data[:count_end:2] == count_data[1:count_end:2]):
                raise AlignmentException('Handshake length data corrupted')
        elif count_shape[0] < n_handshake_ints:
            # have handshake size and remaining is handshake item, not inverted
            if not np.all(
                    count_data[:count_end:2] == count_data[1:count_end:2]):
                raise AlignmentException('Remaining handshake data corrupted')
        else:
            # remaining is regular data, first val is not inverted, the rest is
            if count_end:
                # have at least 2 values (count_end is multiple of 2)
                if count_data[0] != count_data[1]:
                    raise AlignmentException(
                        'Remaining handshake data corrupted')
            if not np.all((count_data[2:count_end:2] ^ count_bits) & count_bits
                          == count_data[3:count_end:2]):
                raise AlignmentException(
                    'Remaining inverted handshake data corrupted')

    def get_handshake(
            self, count: np.ndarray, contains_sub_frames, n_sub_frames):
        """Takes the ``count`` data as returned by :meth:`get_count_ints` and
        extracts the handshake bytes from it.

        It returns a 4-tuple of ``(handshake_data, handshake_len,
        n_handshake_ints, n_config_frames)``.

        ``handshake_data`` is like :attr:`handshake_data`, ``handshake_len``
        is like :attr:`expected_handshake_len`, ``n_handshake_ints`` is the
        number of ints in the handshake, ``n_config_frames`` is the total number
        of configuration frames including the initial count and the handshake.

        If handshake is incomplete it returns empty bytes and all zeros for the
        numbers.
        """
        if len(count):
            n_config = int(count[0])
            # sanity check
            if n_config > 50:
                raise AlignmentException(
                    f'Got too many ({n_config}) handshake numbers')

            # it's in little endian so it doesn't need anything special
            # but we need to remove extra bytes given we use 64 bit (the max)
            config = b''.join(
                [b.tobytes()[:self.n_bytes_per_int]
                 for b in count[1:1 + n_config]])
            n_config_frames = (n_config + 1) * 2 * self.n_parts_per_int
            if contains_sub_frames:
                n_config_frames *= n_sub_frames
            return config, n_config * self.n_bytes_per_int, n_config + 1, \
                n_config_frames

        return b'', 0, 0, 0

    def check_missing_frames(
            self, short_count_data, contains_sub_frames, n_sub_frames):
        """Given the short counter data it checks that no frames was missed
        since it should be contiguous.
        """
        short = short_count_data
        if contains_sub_frames:
            short = short[::n_sub_frames]
        max_shot_val = 2 ** len(self.short_count_indices)

        i = short[0]
        for k, val in enumerate(short):
            if i != val:
                raise AlignmentException(f'Skipped a frame at frame {k}')
            i += 1
            if i == max_shot_val:
                i = 0


class MCSDigitalData(DigitalDataStore):
    """Parses the MCS data recorded into the MCS file.

    In addition to parsing the 24-bits for each frame, for the MCS we first need
    to collapse the data because it's sampling at a much higher rate than the
    Ceed (projector) is generating data (see :meth:`reduce_samples`).

    Finally, as opposed to the Ceed data that is separately stored for each
    experiment, the MCS data contains data for many experiments, which needs to
    be individually parsed (see :ref:`parse-ceed-mcs`).

    The per-experiment parsed data is stored in :attr:`experiments`.
    """

    data_indices_start: np.ndarray = None
    """An array of indices into the raw :attr:`DigitalDataStore.data`
    indicating the start of each (potential) projector frame. We use the clock
    signal to compute this.

    This lets us find align the start of a frame with e-phys data for the frame.
    """

    data_indices_end: np.ndarray = None
    """The same as :attr:`data_indices_start`, but it's the index of the last
    sample of the (potential) frame.

    It is the same length as :attr:`data_indices_start`, because each start
    index has a end index, even if it's single sample long, in which case they
    point to the same index.
    """

    experiments: Dict[
        bytes,
        List[Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]]
    ] = {}
    """A dict of potential experiments found in the MCS parsed
    :attr:`DigitalDataStore.data`.

    Each experiment is uniquely identified by the handshake key Ceed sends
    at the start of that experiment. This key is parsed from
    :attr:`DigitalDataStore.data` for each experiment found in the data,
    ignoring experiments that are too short (a couple of frames).

    Typically the handshake is 16 bytes of data. When
    :attr:`~ceed.view.controller.ViewControllerBase.pad_to_stage_handshake` is
    False or if the experiment was ended early, an experiment could still be
    short enough so that the full handshake was not transmitted. So, e.g.
    you may have two experiments whose unique keys are `"ABCD"` and `"ABGH"`,
    respectively, but only `"AB"` was transmitted before the experiment ended.

    Therefore, each key in the dict maps to a list of experiments, in case
    a single key (`"AB"`) maps to multiple experiments in the MCS data. The
    user would then need to disambiguate between them.

    Each item in the value is a 5-tuple of ``(start, end, handshake_len,
    count_data, count)`` specific to the experiment. ``start`` and ``end``
    are the slice from :attr:`data_indices_start` and :attr:`data_indices_end`
    for the experiment. ``count_data`` is the slice from
    :attr:`DigitalDataStore.count_data` for the experiment and ``count`` is the
    :meth:`DigitalDataStore.get_count_ints` parsed ``count`` from that.
    ``handshake_len`` is the the corresponding value from
    :meth:`DigitalDataStore.get_handshake`, parsed from ``count``.

    .. note::

        This is only populated for experiments created by Ceed versions
        greater than ``1.0.0.dev0``.
    """

    def parse(
            self, ceed_version, data, t_start: datetime.datetime, f,
            find_start_from_ceed_time=False,
            estimated_start: datetime.datetime = None,
            pre_estimated_start: float = 0):
        """Parses the MCS data into the individual experiments.

        :param ceed_version: The file's Ceed version string.
        :param data: the upto 24-bit digital data recorded by MCS at the MCS
            sampling rate (>> the Ced rate) along with the electrode data.
            It is saved as :attr:`DigitalDataStore.data`.
        :param t_start: The date/time that corresponds to the first data sample
            index in ``data``. That's when the MCS data recording started.
        :param f: The MCS recording sampling rate.
        :param find_start_from_ceed_time: Whether to locate the Ceed experiment
            in the MCS data using a Ceed time estimate or by finding the
            handshaking pattern in the digital data. The time based approach
            is not well tested, but can tuned if handshaking is not working.
        :param estimated_start: The estimated time when the Ceed experiment
            started.
        :param pre_estimated_start: A fudge factor for ``estimated_start``. We
            look for the experiment by ``pre_estimated_start`` before
            ``estimated_start``.
        """
        self._parse_components(data)
        self.reduce_samples(
            t_start, f, find_start_from_ceed_time,
            estimated_start, pre_estimated_start)

        if ceed_version == '1.0.0.dev0':
            return
        self.parse_experiments()

    def reduce_samples(
            self, t_start: datetime.datetime, f,
            find_start_from_ceed_time=False,
            estimated_start: datetime.datetime = None,
            pre_estimated_start: float = 0):
        """Reduces the data from multiple samples per-frame, to one sample per
        frame, using the clock. See :meth:`parse`.

        Ceed (projector) generates data at about 120Hz, while MCS records data
        at many thousands of hertz. So each Ceed frame is saved repeatedly over
        many samples. This collapses it into a single sample per frame.

        We can do this because Ceed toggles the clock for each frame.

        Once reduced, it extracts the clock, short and long counter data and
        saves them into the class properties ready for use by the super class
        methods.
        """
        # data is already converted to normal lower bits
        clock_data = self.clock_data
        short_count_data = self.short_count_data
        count_data = self.count_data

        offset = 0
        if find_start_from_ceed_time:
            offset = (estimated_start - t_start).total_seconds() - \
                float(pre_estimated_start)
            if offset < 0:
                raise ValueError(
                    'Ceed data is not in the mcs data, given the offset')

            offset = int(offset * f)
            clock_data = clock_data[offset:]
            short_count_data = short_count_data[offset:]
            count_data = count_data[offset:]

        # should have at least 10 samples. At 5k sampling rate it's reasonable
        if len(clock_data) < 10:
            raise TypeError(
                'There is not enough data in the mcs file to be able to align '
                'with Ceed')

        clock_change = np.argwhere(clock_data[1:] - clock_data[:-1]).squeeze()
        # indices in data where value is different from last (including 0)
        idx_start = np.array([0], dtype=clock_change.dtype)
        # indices in data where next value is different (including last value)
        idx_end = np.array([len(clock_data) - 1], dtype=clock_change.dtype)
        if len(clock_change):
            idx_start = np.concatenate((idx_start, clock_change + 1))
            idx_end = np.concatenate((clock_change, idx_end))

        # take value after clock changes
        indices = np.minimum(idx_end - idx_start, 1) + idx_start

        # start at the
        s = 0 if clock_data[0] else 1
        indices = indices[s:]
        idx_start = idx_start[s:]
        idx_end = idx_end[s:]

        # indices in the original data
        self.data_indices_start = idx_start + offset
        self.data_indices_end = idx_end + offset
        # condensed data
        self.clock_data = clock_data[indices]
        self.short_count_data = short_count_data[indices]
        self.count_data = count_data[indices]

    def parse_experiments(self):
        """Given the data that has been reduced and split into the components
        with :meth:`reduce_samples`, it extracts the individual experiments
        into :attr:`experiments`.
         """
        # assuming the experiments recorded have at least two good frames,
        # otherwise we can't estimate expected frame size
        if self.n_parts_per_int <= 1:
            raise NotImplementedError(
                'Must break counter int into at least two parts so we can '
                'locate clock inverted values')

        max_shot_val = 2 ** len(self.short_count_indices)
        count_data_full = self.count_data
        short_count_data_full = self.short_count_data
        clock_data_full = self.clock_data
        start = self.data_indices_start
        end = self.data_indices_end
        diff = end - start
        med = np.median(diff)
        # each experiment is proceeded by 30-50 blank frames, so 10 is safe.
        # And we should never skip 10+ frames sequentially in a stable system
        breaks = np.nonzero(diff >= (10 * med))[0]

        experiments = self.experiments = defaultdict(list)
        start_i = 0
        for break_i in breaks:
            s = start_i
            # the long frame is included in last experiment. If long frame is
            # clock low, first frame of next exp is high. If it's high, clock
            # goes low for some frames and then high, so high frame will be
            # first short frame
            e = break_i + 1
            # get section of this possible experiment
            count_data = count_data_full[s:e]
            short_count_data = short_count_data_full[s:e]
            clock_data = clock_data_full[s:e]
            start_i = e

            # need some data to work with
            if len(count_data) < 4:
                continue
            # need to start high
            if not clock_data[0]:
                count_data = count_data[1:]
                short_count_data = short_count_data[1:]
                s += 1

            try:
                # use short counter to see if missing frames, exclude final
                self.check_missing_frames(short_count_data[:-1], False, 1)

                # we don't drop last frame, but the frame extends too long post
                # experiment (i.e. last was clock low and it stayed clock low
                # until next experiment). And last item may be spurious
                end[e - 1] = start[e - 1] + med

                # chop off partial ints and get full ints, it's ok if last value
                # is spurious
                count, count_2d, count_inverted_2d = self.get_count_ints(
                    count_data, False, 1)
                # get handshake from full ints. Last val could be spurious, so
                # if it's part of the handshake, handshake is not complete
                handshake_data, handshake_len, n_handshake_ints, \
                    n_config_frames = self.get_handshake(count, False, 1)
                # check that full and partial ints match
                self.check_counter_consistency(
                    count_data, count_2d, count_inverted_2d, n_handshake_ints,
                    False, 1, True)
            except AlignmentException:
                continue

            if not handshake_len:
                continue

            # the last count or handshake value could be spurious, but then it
            # won't match, which is ok because we need the full handshake and
            # when searching for handshake we anyway chop of end until empty
            # or found
            experiments[handshake_data].append((
                start[s:e], end[s:e], handshake_len, count_data, count))


class CeedDigitalData(DigitalDataStore):
    """Parses the Ceed data for an individual experiment as recorded in the
    Ceed H5 file.
    """

    def parse(
            self, ceed_version, frame_bits: np.ndarray,
            frame_counter: np.ndarray, start_t: datetime.datetime,
            n_sub_frames: int, rendered_frames: np.ndarray
    ) -> None:
        """Parses the Ceed data into the class properties from the raw data.

        :param ceed_version: The Ceed version string.
        :param frame_bits: Array containing the 24 bits sent to MCS for each
            Ceed frame.
        :param frame_counter: Array containing the global Ceed counter
            corresponding to each frame. This array includes frames that are
            not actually rendered but dropped by Ceed.
        :param start_t: The date/time when the experiment started.
        :param n_sub_frames: The number of sub-frames in each frame (e.g. for
            quad modes it's 4 or 12).
        :param rendered_frames: Logical array of the same size as
            ``frame_counter`` but only those frames that were rendered are True.
        """
        if ceed_version == '1.0.0.dev0':
            self.parse_data_v1_0_0_dev0(frame_bits)
        else:
            self.parse_data(
                frame_bits, frame_counter, start_t, n_sub_frames,
                rendered_frames)

    def parse_data_v1_0_0_dev0(self, data: np.ndarray) -> None:
        """Parses the data (from :meth:`parse`) for Ceed version ``1.0.0.dev0``.
        """
        self._parse_components(data)

    def parse_data(
            self, frame_bits: np.ndarray, frame_counter: np.ndarray,
            start_t: datetime.datetime, n_sub_frames: int,
            rendered_frames: np.ndarray
    ) -> None:
        """Parses the data (from :meth:`parse`) for Ceed versions greater than
        ``1.0.0.dev0``.
        """
        # the last frame(s) may or may not have been rendered (e.g. with
        # sub-frames, all the sub-frames may not have been rendered)
        contains_sub_frames = True
        self._parse_components(frame_bits[rendered_frames])

        # use short counter to see if missing frames
        self.check_missing_frames(
            self.short_count_data, contains_sub_frames, n_sub_frames)
        # chop off partial ints and get full ints
        count, count_2d, count_inverted_2d = self.get_count_ints(
            self.count_data, contains_sub_frames, n_sub_frames)
        # get handshake from full ints
        handshake_data, handshake_len, n_handshake_ints, n_config_frames = \
            self.get_handshake(count, contains_sub_frames, n_sub_frames)
        # check that full and partial ints match
        self.check_counter_consistency(
            self.count_data, count_2d, count_inverted_2d, n_handshake_ints,
            contains_sub_frames, n_sub_frames, False)

        self.handshake_data = handshake_data
        self.expected_handshake_len = handshake_len
        self.counter = frame_counter[rendered_frames].copy()


class CeedMCSDataMerger:
    """Merges a MCS file containing electrode and digital data recorded during
    a Ceed experiment into a new Ceed file that contains both the original
    Ceed and MCS data.

    See :ref:`merging-ceed-mcs` for an example.
    """

    ceed_filename: str = ''
    """The H5 filename of the file containing the Ceed data.
    """

    ceed_global_config = {}
    """Dict containing the global ``app_settings`` found in the Ceed file.

    This the global app config as present when the file was last saved as
    opposed to the app settings of a specific experiment as stored in
    :attr:`ceed_config_orig`.
    """

    ceed_version: str = ''
    """The Ceed version string read from the Ceed file.
    """

    ceed_config_orig = {}
    """Dict of the ``app_settings`` used for the opened experiment.

    It is updated for each experiment read with
    :meth:`read_ceed_experiment_data`.
    """

    ceed_data = {}
    """Dict with the raw Ceed data for the opened experiment.

    It is updated for each experiment read with
    :meth:`read_ceed_experiment_data`.
    """

    ceed_data_container: Optional[CeedDigitalData] = None
    """The :class:`CeedDigitalData` used to parse the Ceed data.
    """

    mcs_filename: str = ''
    """The H5 filename of the file containing the MCS data.
    """

    mcs_dig_data: np.ndarray = None
    """The raw array containing the digital (16-bits) data recorded by MCS
    at the same frequency and time as the electrode data.
    """

    mcs_dig_config = {}
    """MCS configuration data read from the MCS H5 file.
    """

    mcs_data_container: Optional[MCSDigitalData] = None
    """The :class:`MCSDigitalData` used to parse the MCS data.
    """

    def __init__(self, ceed_filename, mcs_filename):
        self.ceed_filename = ceed_filename
        self.mcs_filename = mcs_filename

    @property
    def n_sub_frames(self):
        """The number of sub-frames in each Ceed frame (e.g. for quad modes
        it's 4 or 12).

        It's specific to the current experiment.
        """
        video_mode = self.ceed_config_orig['view']['video_mode']
        n_sub_frames = 1
        if video_mode == 'QUAD4X':
            n_sub_frames = 4
        elif video_mode == 'QUAD12X':
            n_sub_frames = 12
        return n_sub_frames

    def get_experiment_numbers(self, ignore_list=None) -> List[str]:
        """Returns list of experiments in the :attr:`ceed_filename`.

        ``ignore_list``, if provided, is a list of experiment numbers to skip.

        Each experiment name is a number, represented as a str.

        This method is very light and does not load the data, so it can be
        called before :meth:`read_ceed_data`.
        """
        from ceed.storage.controller import CeedDataWriterBase
        nix_file = nix.File.open(self.ceed_filename, nix.FileMode.ReadOnly)
        try:
            names = CeedDataWriterBase.get_blocks_experiment_numbers(
                nix_file.blocks, ignore_list)
        finally:
            nix_file.close()
        return names

    def read_mcs_data(self):
        """Reads the MCS digital data and metadata from the file and updates the
        corresponding properties.
        """
        filename = self.mcs_filename
        self.mcs_data_container = None

        data = McsPy.McsData.RawData(filename)
        if not len(data.recordings):
            raise Exception('There is no data in {}'.format(filename))
        if len(data.recordings) > 1:
            raise Exception('There is more than one recording in {}'.
                            format(filename))
        t_start = data.date

        chan = f = None
        for stream in data.recordings[0].analog_streams:
            if 128 in data.recordings[0].analog_streams[stream].channel_infos:
                chan = stream
                f = data.recordings[0].analog_streams[stream].\
                    channel_infos[128].sampling_frequency
                f = f.m_as(ureg.hertz)
                break
        if chan is None:
            raise Exception('Did not find digital data channel')

        self.mcs_dig_data = np.array(
            data.recordings[0].analog_streams[chan].channel_data).squeeze()
        self.mcs_dig_config = {'t_start': t_start, 'f': f}

    def read_ceed_data(self):
        """Reads the overall Ceed metadata from the file and updates the
        corresponding properties.

        After loading the overall data, use :meth:`read_ceed_experiment_data`
        to load a specific experiment.
        """
        from ceed.analysis import read_nix_prop
        filename = self.ceed_filename
        self.ceed_data_container = None

        f = nix.File.open(filename, nix.FileMode.ReadOnly)
        try:
            config_data = {}
            for prop in f.sections['app_config'].props:
                config_data[prop.name] = yaml_loads(read_nix_prop(prop))
        finally:
            f.close()

        self.ceed_global_config = config_data['app_settings']
        self.ceed_version = config_data['ceed_version']

    def read_ceed_experiment_data(self, experiment: Union[str, int]):
        """Reads the data and metadata for the experiment and updates the
        corresponding properties.

        :param experiment: The experiment to read. It's a number (or string
            representing the number) for the experiment, as would be returned
            by :meth:`get_experiment_numbers`.
        """
        from ceed.storage.controller import CeedDataWriterBase
        from ceed.analysis import read_nix_prop
        if not self.ceed_global_config:
            raise TypeError(
                'Global ceed data not read. Please first call read_ceed_data')

        filename = self.ceed_filename
        f = nix.File.open(filename, nix.FileMode.ReadOnly)

        block = f.blocks[
            CeedDataWriterBase.get_experiment_block_name(experiment)]
        section = block.metadata
        start_t = datetime.datetime(1970, 1, 1) + \
            datetime.timedelta(seconds=block.created_at)

        metadata = {}
        try:
            try:
                config = section.sections['app_config']
            except KeyError as exc:
                raise KeyError(
                    'Did not find config in experiment info for experiment {}'.
                    format(experiment)) from exc

            for prop in config:
                metadata[prop.name] = yaml_loads(read_nix_prop(prop))
            self.ceed_config_orig = metadata['app_settings']
            # ceed_config_orig must be set to read n_sub_frames
            n_sub_frames = self.n_sub_frames
            skip = self.ceed_config_orig['view'].get(
                'skip_estimated_missed_frames', False)

            if not block.data_arrays['frame_bits'].shape or \
                    not block.data_arrays['frame_bits'].shape[0]:
                raise Exception('Experiment {} has no data'.format(experiment))

            frame_bits = np.asarray(block.data_arrays['frame_bits']).squeeze()
            frame_counter = np.asarray(
                block.data_arrays['frame_counter']).squeeze()

            # rendered_counter is multiples of n_sub_frames, starting from
            # n_sub_frames. Missed frames don't have number in rendered_counter
            rendered_counter = np.asarray(
                block.data_arrays['frame_time_counter']).squeeze()
            if skip:
                count_indices = np.arange(1, 1 + len(frame_counter))
                found = rendered_counter[:, np.newaxis] - \
                    np.arange(n_sub_frames)[np.newaxis, :]
                found = found.reshape(-1)
                rendered_frames = np.isin(count_indices, found)
            else:
                if not np.all(
                        rendered_counter == np.arange(
                            n_sub_frames, len(frame_counter) + 1, n_sub_frames)
                ):
                    raise ValueError(
                        'Some frames were not rendered and skipped')

                rendered_frames = np.ones(len(frame_counter), dtype=np.bool)

        except Exception:
            f.close()
            raise
        else:
            f.close()

        self.ceed_data = {
            'frame_bits': frame_bits, 'frame_counter': frame_counter,
            'start_t': start_t, 'rendered_frames': rendered_frames}

    def create_or_reuse_ceed_data_container(self) -> CeedDigitalData:
        """Ensures that the :attr:`ceed_data_container` is ready to be used
        to parse the data.
        """
        config = self.ceed_config_orig['serializer']
        short_count_indices = config['short_count_indices']
        count_indices = config['count_indices']
        clock_index = config['clock_idx']
        counter_bit_width = config['counter_bit_width']

        if self.ceed_data_container is None:
            self.ceed_data_container = CeedDigitalData(
                short_count_indices, count_indices, clock_index,
                counter_bit_width)
        elif not self.ceed_data_container.compare(
                short_count_indices, count_indices, clock_index):
            raise ValueError('Ceed-MCS hardware mapping has changed in file')
        return self.ceed_data_container

    def create_or_reuse_mcs_data_container(self) -> MCSDigitalData:
        """Ensures that the :attr:`mcs_data_container` is ready to be used
        to parse the data.
        """
        config = self.ceed_global_config['serializer']
        ceed_mcs_map = config['projector_to_aquisition_map']

        short_count_indices = [
            ceed_mcs_map[i] for i in config['short_count_indices']]
        count_indices = [ceed_mcs_map[i] for i in config['count_indices']]
        clock_index = ceed_mcs_map[config['clock_idx']]
        counter_bit_width = config['counter_bit_width']

        if self.mcs_data_container is None:
            self.mcs_data_container = MCSDigitalData(
                short_count_indices, count_indices, clock_index,
                counter_bit_width)
        elif not self.mcs_data_container.compare(
                short_count_indices, count_indices, clock_index):
            raise ValueError('Ceed-MCS hardware mapping has changed in file')
        return self.mcs_data_container

    def parse_ceed_experiment_data(self):
        """Parses the Ceed experiment data previously read with
        :meth:`read_ceed_experiment_data`.
        """
        if not self.ceed_data:
            raise TypeError(
                'Ceed experiment data not read. Please first call '
                'read_ceed_experiment_data')

        n_sub_frames = self.n_sub_frames
        self.create_or_reuse_ceed_data_container()
        self.ceed_data_container.parse(
            self.ceed_version, **self.ceed_data, n_sub_frames=n_sub_frames)

    def parse_mcs_data(
            self, find_start_from_ceed_time: bool = False,
            pre_estimated_start: float = 0,
            estimated_start: datetime.datetime = None):
        """Parses the MCS data previously read with :meth:`read_mcs_data`.

        :param find_start_from_ceed_time: Whether to locate the Ceed experiment
            in the MCS data using a Ceed time estimate or by finding the
            handshaking pattern in the digital data. The time based approach
            is not well tested, but can tuned if handshaking is not working.
        :param estimated_start: The estimated time when the Ceed experiment
            started. You can get this with ``self.ceed_data['start_t']``.
        :param pre_estimated_start: A fudge factor for ``estimated_start``. We
            look for the experiment by ``pre_estimated_start`` before
            ``estimated_start``.
        """
        if not self.mcs_dig_config:
            raise TypeError(
                'MCS data not read. Please first call read_mcs_data')
        if not self.ceed_global_config:
            raise TypeError(
                'Global ceed data not read. Please first call read_ceed_data')

        self.create_or_reuse_mcs_data_container()
        self.mcs_data_container.parse(
            self.ceed_version, self.mcs_dig_data,
            self.mcs_dig_config['t_start'], self.mcs_dig_config['f'],
            find_start_from_ceed_time=find_start_from_ceed_time,
            pre_estimated_start=pre_estimated_start,
            estimated_start=estimated_start)

    def get_alignment(self) -> np.ndarray:
        """After reading and parsing the Ceed and MCS data you can compute
        the alignment between the current Ceed experiment and the MCS file.

        This returns an array of indices, where each item in the array is an
        index into the raw MCS electrode data and corresponds to a Ceed frame.
        The index is the start index in the electrode data corresponding to when
        the ith Ceed frame was beginning to be displayed.
        """
        if self.ceed_version == '1.0.0.dev0':
            return self._get_alignment_v1_0_0_dev0(True)
        return self._get_alignment(True)

    def _get_alignment(self, search_uuid=True) -> np.ndarray:
        if not search_uuid:
            raise NotImplementedError

        ceed_ = self.ceed_data_container
        mcs = self.mcs_data_container
        handshake = ceed_.handshake_data
        if not handshake:
            raise AlignmentException(
                'Cannot find experiment - no Ceed experiment ID parsed')

        if handshake not in mcs.experiments:
            if not mcs.experiments:
                raise AlignmentException(
                    'Cannot find any experiment in the MCS parsed data')

            while handshake and handshake not in mcs.experiments:
                handshake = handshake[:-1]

            if not handshake:
                raise AlignmentException(
                    'Cannot find experiment in the MCS parsed data')

        experiments = mcs.experiments[handshake]
        if len(experiments) != 1:
            raise AlignmentException(
                'Found more than one matching experiment in MCS data, '
                'experiment was likely stopped before the full Ceed-MCS '
                'handshake completed')

        # the last count or handshake value could be spurious, but then it
        # won't match, which is ok because we need the full handshake
        start, end, handshake_len, count_data, count = experiments[0]
        # n_sub_frames can change between experiments
        n_sub_frames = self.n_sub_frames
        # count_data is same for all sub-frames, but counter increments
        ceed_count_data = ceed_.count_data

        # ceed counter contains an item for each frame and sub-frame
        assert not len(ceed_count_data) % n_sub_frames
        # mcs only sees frames, because sub-frames are all the same
        ceed_count_data_main_frames = ceed_count_data[::n_sub_frames]
        n_ceed = len(ceed_count_data_main_frames)
        n_mcs = len(count_data)
        assert n_mcs

        if n_mcs < n_ceed:
            raise AlignmentException(
                'MCS missed some ceed frames, cannot align')
        if n_mcs > n_ceed + 1:
            raise AlignmentException(
                'MCS read frames that ceed did not send, cannot align')

        if n_mcs != n_ceed:
            # last frame could be spurious on the mcs side
            n_mcs -= 1

        count_data = count_data[:n_mcs]
        if not np.all(ceed_count_data_main_frames == count_data):
            raise AlignmentException(
                'Counter data itemds does not match between Ceed and MCS')

        start = start[:n_mcs]
        end = end[:n_mcs]
        if n_sub_frames == 1:
            return start

        n_frames = end - start + 1
        n_frames = n_frames[:, np.newaxis] / n_sub_frames
        split_frames = np.arange(n_sub_frames)[np.newaxis, :]
        start = np.round(split_frames * n_frames + start[:, np.newaxis])
        start = np.asarray(start, dtype=np.int64).reshape(-1)

        return start

    def _get_alignment_v1_0_0_dev0(self, search_uuid=True) -> np.ndarray:
        ceed_ = self.ceed_data_container
        mcs = self.mcs_data_container

        s = 0
        if search_uuid:
            # n is number of frames we need to send uuid
            n = int(ceil(32 / float(len(ceed_.count_indices)))) * 5

            # following searches for the uuid in the mcs data
            strides = mcs.count_data.strides + (mcs.count_data.strides[-1], )
            shape = mcs.count_data.shape[-1] - n + 1, n
            strided = np.lib.stride_tricks.as_strided(
                mcs.count_data, shape=shape, strides=strides)
            res = np.all(strided == ceed_.count_data[:n], axis=1)
            indices = np.mgrid[0:len(res)][res]

            if not len(indices):
                raise AlignmentException('Could not find alignment')
            if len(indices) > 1:
                raise Exception(
                    'Found multiple Ceed-mcs alignments ({})'.format(indices))

            s = indices[0]

        e = len(ceed_.clock_data) - 1 + s
        mcs_indices = mcs.data_indices_start[s:e]

        if np.all(mcs.clock_data[s:e] == ceed_.clock_data[:-1]) and \
            np.all(mcs.short_count_data[s:e] == ceed_.short_count_data[:-1]) \
                and np.all(mcs.count_data[s:e] == ceed_.count_data[:-1]):
            return mcs_indices

        raise AlignmentException('Could not align the data')

    def estimate_skipped_frames(self, ceed_mcs_alignment):
        """Given an alignment from :meth:`get_alignment`, it returns
        information about the Ceed frames to help compute whether Ceed dropped
        any frames, as observed by the recorded MCS data.

        See :meth:`get_skipped_frames_summary` and
        :meth:`~ceed.analysis.CeedDataReader.compute_long_and_skipped_frames`.
        """
        from ceed.analysis import CeedDataReader
        return CeedDataReader.compute_long_and_skipped_frames(
            self.n_sub_frames, self.ceed_data['rendered_frames'],
            ceed_mcs_alignment)

    def get_skipped_frames_summary(self, ceed_mcs_alignment, experiment_num):
        """Given an alignment from :meth:`get_alignment` and the experiment
        number, it returns a string that when printed will display some
        information about any dropped Ceed frames as well as overall information
        about the alignment.

        There are 8 numbers in the summary.

        1.   The experiment number.
        2-3. The start index and end index in the MCS electrode data
             corresponding to the experiment.
        4.   The number of electrode data samples during the experiment.
        5.   The number of Ceed frames that went long (i.e. the number of frames
             whose data was not the next frame, but an older frame that the GPU
             displayed again because the CPU didn't update in time.
        6.   The number of major frames Ceed dropped.
        7.   In parenthesis, the number of total frames, including sub-frames
             Ceed dropped.
        8.   The maximum delay in terms of frames between Ceed repeating a frame
             because the CPU was too slow, until Ceed realized it needs to drop
             a frame to compensate. I.e. this is the largest number of bad
             frames Ceed ever displayed before correcting.
        """
        mcs_long_frames, mcs_frame_len, ceed_skipped, ceed_skipped_main, \
            largest_bad = self.estimate_skipped_frames(ceed_mcs_alignment)
        num_long = sum(i - 1 for i in mcs_frame_len)
        main_skipped = len(ceed_skipped_main)
        skipped = len(ceed_skipped)

        return (
            f'Aligned experiment {experiment_num: >2}. '
            f'MCS: [{ceed_mcs_alignment[0]: 10} - '
            f'{ceed_mcs_alignment[-1]: 10}] '
            f'({len(ceed_mcs_alignment): 7} frames). {num_long: 3} slow, '
            f'{skipped: 3} ({main_skipped: 2}) dropped. Max {largest_bad} bad')

    def merge_data(
            self, filename: str, alignment_indices: Dict[str, np.ndarray],
            notes='', notes_filename=None):
        """Takes the Ceed and MCS data files and copies them over to a new
        Ceed H5 file. It also adds the alignment data for all the experiments
        as computed with :meth:`get_alignment` and optional notes and stores it
        in the new file.

        :param filename: The target file to create.
        :param alignment_indices: Dict whose keys are experiment numbers and
            whose values is the alignment for that experiment as computed by
            :meth:`get_alignment`.
        :param notes: Optional notes string to be added to the new file.
        :param notes_filename: A optional filename, which if provided contains
            more notes that is appended to ``notes`` and added to the file.
        """
        if os.path.exists(filename):
            raise Exception('{} already exists'.format(filename))

        if notes_filename and os.path.exists(notes_filename):
            with open(notes_filename, 'r') as fh:
                lines = fh.read()

            if notes and lines:
                notes += '\n'
            notes += lines

        mcs_f = McsPy.McsData.RawData(self.mcs_filename)
        if not len(mcs_f.recordings):
            raise Exception('There is no data in {}'.format(self.mcs_filename))
        if len(mcs_f.recordings) > 1:
            raise Exception('There is more than one recording in {}'.
                            format(self.mcs_filename))

        copy2(self.ceed_filename, filename)
        f = nix.File.open(filename, nix.FileMode.ReadWrite)

        if 'app_logs' not in f.sections:
            f.create_section('app_logs', 'log')
            f.sections['app_logs']['notes'] = ''

        if f.sections['app_logs']['notes'] and notes:
            f.sections['app_logs']['notes'] += '\n'
        f.sections['app_logs']['notes'] += notes

        block = f.create_block(
            'ceed_mcs_alignment', 'each row, r, contains the sample index '
            'in the mcs data corresponding with row r in ceed data. This is '
            'the index at which the corresponding Ceed frame was displayed')
        for exp, indices in alignment_indices.items():
            block.create_data_array(
                'experiment_{}'.format(exp), 'experiment_{}'.format(exp),
                data=indices)

        streams = mcs_f.recordings[0].analog_streams
        num_channels = 0
        for stream in streams.values():
            if 128 in stream.channel_infos:
                num_channels += 1
            elif 0 in stream.channel_infos:
                num_channels += len(stream.channel_infos)

        pbar = tqdm(
            total=num_channels, file=sys.stdout, unit_scale=1,
            unit='data channels')

        block = f.create_block('mcs_data', 'mcs_raw_experiment_data')
        block.metadata = section = f.create_section(
            'mcs_metadata', 'Associated metadata for the mcs data')
        for stream_id, stream in streams.items():
            if 128 in stream.channel_infos:
                pbar.update()
                block.create_data_array(
                    'digital_io', 'digital_io',
                    data=np.array(stream.channel_data).squeeze())
            elif 0 in stream.channel_infos:
                for i in stream.channel_infos:
                    info = stream.channel_infos[i].info
                    pbar.update()

                    elec_sec = section.create_section(
                        'electrode_{}'.format(info['Label']),
                        'metadata for this electrode')
                    for key, val in info.items():
                        if isinstance(val, np.generic):
                            val = val.item()
                        elec_sec[key] = yaml_dumps(val)

                    freq = stream.channel_infos[i].sampling_frequency
                    freq = freq.m_as(ureg.hertz).item()
                    ts = stream.channel_infos[i].sampling_tick
                    ts = ts.m_as(ureg.second).item()

                    elec_sec['sampling_frequency'] = yaml_dumps(freq)
                    elec_sec['sampling_tick'] = yaml_dumps(ts)

                    data = np.array(stream.channel_data[i, :])
                    block.create_data_array(
                        'electrode_{}'.format(info['Label']), 'electrode_data',
                        data=data)

        pbar.close()
        f.close()


if __name__ == '__main__':
    ceed_file = 'ceed_data.h5'
    mcs_file = 'mcs_data.h5'
    output_file = 'ceed_mcs_merged.h5'
    notes = ''
    notes_filename = None

    merger = CeedMCSDataMerger(ceed_filename=ceed_file, mcs_filename=mcs_file)

    merger.read_mcs_data()
    merger.read_ceed_data()
    merger.parse_mcs_data()

    alignment = {}
    for experiment in merger.get_experiment_numbers(ignore_list=[]):
        merger.read_ceed_experiment_data(experiment)
        merger.parse_ceed_experiment_data()

        try:
            align = alignment[experiment] = merger.get_alignment()
            # see get_skipped_frames_summary for meaning of the printed values
            print(merger.get_skipped_frames_summary(align, experiment))
        except Exception as e:
            print(
                "Couldn't align MCS and ceed data for experiment "
                "{} ({})".format(experiment, e))

    merger.merge_data(
        output_file, alignment, notes=notes, notes_filename=notes_filename)
