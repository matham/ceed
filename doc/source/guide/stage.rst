.. _stage-guide:

Ceed stages
-----------

A Ceed :ref:`stage <stage-api>` uses its functions to control the intensity of its
shapes during an experiment.

A stage contains functions, shapes, and children stages. Each experiments runs a root
stage and this stage ticks through its functions and children stages. For every video
frame the current time is computed as the frame number multiplied by the period of
the video frame rate. This time is passed to the functions and sub-stages which
set their corresponding shapes for that frame from their functions' return value.

Creating stages
---------------

Ceed comes with a :ref:`pre-defined <stage-factory-plugin>` stage with various parameters,
but a :ref:`plugin <stage-plugin>` can create custom stage classes and add them to the
available stages list. Additionally, you can customize stages in the stage pane to make it
available for re-use in other stages or to be run directly.

This video shows multiple how to create a stage and name it, how to drag a sequence of
functions into it and how to drag a shape and shape group into it.

.. video:: ../media/guide/stage_create.webm

Previewing stage
----------------

Once we have a stage with functions and shapes, when run as an experiment Ceed will go
through these functions sequentially until each is done before moving to the next,
meanwhile it applies the intensity from the function for all its shapes for each frame.

You can preview the exact temporal sequence of intensity that each shape will experience.

Select the stage to preview, the sampling rate (it should match the projector rate for
best accuracy, but it could take a while to pre-compute the entire stage) and refresh
to compute.

You can select which color channel (red, green, blue) to display and zoom to specific
time intervals to see its intensity.

For example in this video, only the blue channel and only three shapes have intensity,
the others have zero intensity (black). The temporal pattern is a cosine ("Slow")
for 30 seconds followed by a constant intensity for a few seconds (the duration and
intensity is randomized).

.. video:: ../media/guide/stage_graph.webm

Modifying functions
-------------------

Rather than modifying the functions in the function pane, you can modify the
function directly in the stage. When originally adding the function, it adds a
reference to the original function. Clocking the T-fork button will replace it
with an editable function.

We can also set the stage to loop over its function sequence, e.g. 3 times in the
video.

.. video:: ../media/guide/stage_mod_function.webm

Running preview stage
---------------------

You can preview a stage by running it in the drawing are and it will play the
exact sequence of intensity as during a real experiment.

Select the stage name and then press the play button to preview. Press stop
to end it (or wait until it's done).

.. video:: ../media/guide/stage_run.webm

:download:`Ceed config of the video. <../media/guide/stage_run.yml>`

Shape color
-----------

You can select the any of the three red, green, and blue color channels to be set
by the function intensity value. The other channels are kept at zero.

.. video:: ../media/guide/stage_run_other_color.webm

:download:`Ceed config of the video. <../media/guide/stage_run_other_color.yml>`

Sub-stages
----------

.. video:: ../media/guide/stage_parallel.webm

:download:`Ceed config of the video. <../media/guide/stage_parallel.yml>`

.. _stage-donut:

Donut stage shapes
------------------

.. video:: ../media/guide/stage_donut.webm

:download:`Ceed config of the video. <../media/guide/stage_donut.yml>`

See :ref:`<func-precision>`
