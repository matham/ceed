ceed Config
===========

:app:

`inspect`: False
 Enables GUI inspection. If True, it is activated by hitting ctrl-e in
 the GUI.


:data:

`backup_interval`: 5.0

`root_path`: 


:function:

`timebase_denominator`: 1
 The denominator of the default timebase. See :attr:`timebase`.

`timebase_numerator`: 1
 The numerator of the default timebase. See :attr:`timebase`.


:player:

`browse_path`: 


:point_gray_cam:

`cam_config_opts`: {}
 The configuration options used to configure the camera after opening.

`cls`: 
 (internal) The string associated with the player source used.
 It is one of ``FFMpeg``, ``RTV``, or ``PTGray`` indicating the camera
 being used.

`estimate_record_rate`: False

`ip`: 
 The ip address of the camera to open. Either :attr:`ip` or
 :attr:`serial` must be provided.

`metadata_play`: None
 (internal) Describes the video metadata of the video player.

`metadata_play_used`: None
 (internal) Describes the video metadata of the video player that is
 actually used by the player.

`metadata_record`: None
 (internal) Describes the video metadata of the video recorder.

`record_directory`: E:\msys64\home\Matthew Einhorn
 The directory into which videos should be saved.

`record_fname`: video{}.avi
 The filename to be used to record the next video.
 If ``{}`` is present in the filename, it'll be replaced with the value of
 :attr:`record_fname_count` which auto increments after every video, when
 used.

`record_fname_count`: 0
 A counter that auto increments by one after every recorded video.
 Used to give unique filenames for each video file.

`serial`: 0
 The serial number of the camera to open. Either :attr:`ip` or
 :attr:`serial` must be provided.


:serializer:

`clock_idx`: 0

`count_indices`: [4, 5]

`counter_bit_width`: 16

`projector_to_aquisition_map`: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

`short_count_indices`: [1, 2, 3]


:video_file_playback:

`cls`: 
 (internal) The string associated with the player source used.
 It is one of ``FFMpeg``, ``RTV``, or ``PTGray`` indicating the camera
 being used.

`dshow_opt`: 
 The camera options associated with :attr:`dshow_true_filename` when
 dshow is used.

`dshow_true_filename`: 
 The real and complete filename of the direct show (webcam) device.

`estimate_record_rate`: False

`file_fmt`: dshow
 The format used to play the video. Can be empty or a format e.g.
 ``dshow`` for webcams.

`icodec`: 
 The codec used to open the video stream with.

`metadata_play`: None
 (internal) Describes the video metadata of the video player.

`metadata_play_used`: None
 (internal) Describes the video metadata of the video player that is
 actually used by the player.

`metadata_record`: None
 (internal) Describes the video metadata of the video recorder.

`play_filename`: 
 The filename of the media being played. Can be e.g. a url etc.

`record_directory`: E:\msys64\home\Matthew Einhorn
 The directory into which videos should be saved.

`record_fname`: video{}.avi
 The filename to be used to record the next video.
 If ``{}`` is present in the filename, it'll be replaced with the value of
 :attr:`record_fname_count` which auto increments after every video, when
 used.

`record_fname_count`: 0
 A counter that auto increments by one after every recorded video.
 Used to give unique filenames for each video file.


:view:

`LED_mode`: RGB
 The LED mode the projector is set to during the experiment.
 Its value is from the :attr:`led_modes`.

`LED_mode_idle`: RGB
 The LED mode the projector is set to before/after the experiment.
 Its value is from the :attr:`led_modes`.

`cam_offset_x`: 0
 The x offset of the background image.

`cam_offset_y`: 0
 The y offset of the background image.

`cam_rotation`: 0
 The rotation angle of the background image.

`cam_scale`: 1.0
 The scaling factor of the background image.

`frame_rate`: 120.0
 The frame rate at which the data is played. This should match the
 currently selected monitor's refresh rate.

`fullscreen`: False
 Whether the second window should run in fullscreen mode. In fullscreen
 mode the window has no borders.

`output_count`: True
 Whether the corner pixel is used to output frame information on the
 PROPixx controller IO pot. If True,
 :class:`ceed.storage.controller.DataSerializer` is used to set the 24 bits
 of the corner pixel.

`preview`: True
 When run, if True, the data is played in the main GUI. When False,
 the data id played on the second window.

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

`video_mode`: RGB
 The current video mode from the :attr:`video_modes`.

