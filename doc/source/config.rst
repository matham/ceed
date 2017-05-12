ceed Config
===========

:app:

`inspect`: False
 Enables GUI inspection. If True, it is activated by hitting ctrl-e in
 the GUI.
 

:data:

`backup_interval`: 5.0
`root_path`: 

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
`cam_offset_x`: 0
`cam_offset_y`: 0
`cam_rotation`: 0
`cam_scale`: 1.0
`frame_rate`: 60.0
`fullscreen`: False
`output_count`: True
`preview`: True
`screen_height`: 1080
`screen_offset_x`: 0
`screen_width`: 1920
`use_software_frame_rate`: True
`video_mode`: RGB
