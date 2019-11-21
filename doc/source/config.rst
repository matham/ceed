CEED Config
===========

The following are the configuration options provided by the app. It can be configured by changing appropriate values in the ``config.yaml`` file. The options default to the default value in the classes configurable by these options.

app
---

`inspect`: False


data
----

`backup_interval`: 5.0

`root_path`: ""


player
------

`player_name`: "ffmpeg"


serializer
----------

`clock_idx`: 2

`count_indices`: [19, 20]

`counter_bit_width`: 32

`projector_to_aquisition_map`: {2: 0, 3: 1, 4: 2, 10: 3, 11: 4, 12: 5, 18: 6, 19: 7, 20: 8}

`short_count_indices`: [3, 4, 10, 11, 12, 18]


view
----

`LED_mode`: "RGB"
 The LED mode the projector is set to during the experiment.
 Its value is from the :attr:`led_modes`.

`LED_mode_idle`: "RGB"
 The LED mode the projector is set to before/after the experiment.
 Its value is from the :attr:`led_modes`.

`cam_transform`: ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

`flip_camera`: False

`flip_projector`: True

`frame_rate`: 120.0
 The frame rate at which the data is played. This should match the
 currently selected monitor's refresh rate.

`fullscreen`: False
 Whether the second window should run in fullscreen mode. In fullscreen
 mode the window has no borders.

`mea_diameter`: 3

`mea_num_cols`: 12

`mea_num_rows`: 12

`mea_pitch`: 20

`mea_transform`: ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

`mirror_mea`: True

`output_count`: True
 Whether the corner pixel is used to output frame information on the
 PROPixx controller IO pot. If True,
 :class:`ceed.storage.controller.DataSerializerBase` is used to set the 24 
 bits of the corner pixel.

`screen_height`: 1080
 The screen height on which the data is played. This is the full-screen
 size.

`screen_offset_x`: 0
 When there are multiple monitors, the window on which the data is played
 is controlled by the position of the screen. E.g. to set it on the right
 screen of two screens, each 1920 pixel wide and with the main screen being
 on the left. Then the :attr:`screen_offset_x` should be set to ``1920``.

`screen_width`: 1920
 The screen width on which the data is played. This is the full-screen
 size.

`use_software_frame_rate`: False
 Depending on the GPU, the software is unable to render faster than the
 GPU refresh rate. In that case, :attr:`frame_rate`, should match the value
 that the GPU is playing at and this should be False.
 If the GPU isn't forcing a frame rate. Then this should be True and
 :attr:`frame_rate` should be the desired frame rate.
 One can tell whether the GPU is forcing a frame rate by setting
 :attr:`frame_rate` to a large value and setting
 :attr:`use_software_frame_rate` to False and seeing what the resultant
 frame rate is. If it isn't capped at some value, e.g. 120Hz, it means that
 the GPU isn't forcing it.

`video_mode`: "RGB"
 The current video mode from the :attr:`video_modes`.

