Ceed functions
==============

A Ceed :ref:`function <function-api>` is how a stage attaches a time-varying intensity
to its shapes.

The stage calls the function or sequence of functions for every video frame with
the current global time and the function returns a scalar value between 0-1, inclusive.
This intensity is used for that stage's shapes for that frame.

Creating functions
------------------

Ceed comes with :ref:`pre-defined <function-factory-plugin>` functions, but a
:ref:`plugin <func-plugin>` can add to the available functions.
Additionally, you can customize functions in the function pane to make it available for
re-use in other function groups or stages.

This video shows multiple ways to create a customized global cosine function from
existing cosine functions.

.. video:: ../media/guide/function_creating.webm

Customizing functions
---------------------

Each function has parameters that describe its output as a function of time. It also has a
duration - the time :ref:`domain where it's valid. <func-domain>` It can also be looped
``n`` times, where the function restarts ``n`` times over its domain until done.

.. video:: ../media/guide/function_customize.webm

Randomizing functions
---------------------

Function parameters can be :ref:`randomized <func-random-param>` so it is resampled before
a new experiment, or once for every loop iteration. It can also be sampled and locked
so all instances tha re-use it share the same randomized values.

As seen in the video, there are multiple distributions to choose from and more can
be added by plugins.

.. video:: ../media/guide/function_random.webm

Function groups
---------------

Functions groups enable iterating through a sequence of functions as if it was a single
function whose duration is the sum of its children functions, recursively.

Initially, when adding or dragging a global function into a group, only a reference to
the function is added. So changing the original function will change the reference
function as well, and the original function cannot be deleted. A reference function is
indicated with the T-junction icon in the video. Clicking it will break the reference
and convert it to a independent function.

Ceed guards from adding a group function to itself or its children, to prevent
infinite recursion as shown at the end of video.

.. video:: ../media/guide/function_group.webm

.. _func-precision:

Function timebase
-----------------

Functions are sampled by the stage at integer multiples of the video frame rate period.
As function duration approaches the period, specifying duration using time in decimal
leads to inaccuracy. Functions therefore support specifying duration as integer multiple
of video frames. E.g. it can be set to be exactly one frame long.

When the function :ref:`timebase <func-timebase>` is specified and non-zero, the duration
is multiplied by the timebase to get the duration in seconds. So when the timebase is
exactly one over the frame rate - the period, a duration of one means one frame.

E.g. in the video, the frame rate is 59.94, or 2997 / 50. Setting the timebase fraction
to 50 / 2997 will allow use to set the duration of each constant function in the group
to one frame. Then, setting the intensity (``a``) of the first child function to zero,
the second one to one, and the group to loop over them 500 times will create a
function that alternates between zero and one intensity for each frame.

The stages and preview graph are explained in the :ref:`stage guide <stage-guide>`.

.. video:: ../media/guide/function_precision.webm

:download:`Ceed config of the video <../media/guide/function_precision.yml>`
