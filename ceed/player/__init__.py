"""Video Player
================

Plays and records media from e.g. a camera or the network.
"""

from os.path import abspath, isdir, dirname, join, exists
import psutil
from ffpyplayer.pic import ImageLoader

from kivy.event import EventDispatcher
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty
from kivy.clock import Clock
from kivy.app import App

from cpl_media.ptgray import PTGrayPlayer, PTGraySettingsWidget
from cpl_media.ffmpeg import FFmpegPlayer, FFmpegSettingsWidget
from cpl_media.thorcam import ThorCamPlayer, ThorCamSettingsWidget
from cpl_media.remote.client import RemoteVideoPlayer, \
    ClientPlayerSettingsWidget
from cpl_media.player import BasePlayer

from cpl_media.recorder import ImageFileRecorder, VideoRecorder, \
    ImageFileRecordSettingsWidget, VideoRecordSettingsWidget
from cpl_media.recorder import BaseRecorder

__all__ = ('CeedPlayer', )


class CeedPlayer(EventDispatcher):
    """Player and recorder that supports playing and recording from multiple
    player and recorder sources made available through :mod:`cpl_media`.

    Through the GUI which of the players is the current :attr:`player`
    and similarly for the :attr:`recorder`.
    """

    _config_props_ = ('player_name', )

    _config_children_ = {
        'ffmpeg': 'ffmpeg_player', 'ptgray': 'ptgray_player',
        'thor': 'thor_player', 'network_client': 'client_player',
        'image_file_recorder': 'image_file_recorder',
        'video_recorder': 'video_recorder',
    }

    ffmpeg_player: FFmpegPlayer = None
    """Player that can playback a video file.
    """

    ffmpeg_settings = None
    """The settings widget used in the GUI to configure the
    :attr:`ffmpeg_player`.
    """

    ptgray_player: PTGrayPlayer = None
    """Player that can play a PointGray camera.
    """

    ptgray_settings = None
    """The settings widget used in the GUI to configure the
    :attr:`ptgray_player`.
    """

    thor_player: ThorCamPlayer = None
    """Player that can play a Thor camera.
    """

    thor_settings = None
    """The settings widget used in the GUI to configure the
    :attr:`thor_player`.
    """

    client_player: RemoteVideoPlayer = None
    """Player that can play a video from a network stream.
    """

    client_settings = None
    """The settings widget used in the GUI to configure the
    :attr:`client_player`.
    """

    player: BasePlayer = ObjectProperty(None, rebind=True)
    """Currently selected player. It is one of :attr:`ffmpeg_player`,
    :attr:`ptgray_player`, :attr:`thor_player`, or :attr:`client_player`.
    """

    player_settings = ObjectProperty(None)
    """The settings widget used in the GUI to configure the currently selected
    player. It is one of :attr:`ffmpeg_settings`,
    :attr:`ptgray_settings`, :attr:`thor_settings`, or :attr:`client_settings`.
    """

    player_name = StringProperty('ffmpeg')
    """The name of the currently selected video player. It is one of "ffmpeg",
    "thor", "ptgray", or "client".
    """

    player_to_raw_name_map = {
        'Webcam/File': 'ffmpeg', 'Network': 'client', 'Thor': 'thor',
        'PointGray': 'ptgray'
    }
    """Maps a user friendly player-type name to the name used with
    :attr:`player_name`.
    """

    player_to_nice_name_map = {v: k for k, v in player_to_raw_name_map.items()}
    """Maps :attr:`player_name` to a user friendly name.
    """

    image_file_recorder: ImageFileRecorder = None
    """Recorder that can record the :attr:`player` video to a image file series.
    """

    image_file_recorder_settings = None
    """The settings widget used in the GUI to configure the
    :attr:`image_file_recorder`.
    """

    video_recorder: VideoRecorder = None
    """Recorder that can record the :attr:`player` video to a video file.
    """

    video_recorder_settings = None
    """The settings widget used in the GUI to configure the
    :attr:`video_recorder`.
    """

    recorder: BaseRecorder = ObjectProperty(None, rebind=True)
    """Currently selected recorder. It is one of :attr:`image_file_recorder` or
    :attr:`video_recorder`.
    """

    recorder_settings = ObjectProperty(None)
    """The settings widget used in the GUI to configure the currently selected
    recorder. It is one of :attr:`image_file_recorder_settings` or
    :attr:`video_recorder_settings`.
    """

    recorder_name = StringProperty('video')
    """The name of the currently selected video recorder. It is one of
    "image_file" or "video".
    """

    recorder_to_raw_name_map = {'Images': 'image_file', 'Video': 'video'}
    """Maps a user friendly recorder-type name to the name used with
    :attr:`recorder_name`.
    """

    recorder_to_nice_name_map = {
        v: k for k, v in recorder_to_raw_name_map.items()}
    """Maps :attr:`recorder_name` to a user friendly name.
    """

    last_image = ObjectProperty(None, allownone=True)
    """The last :attr:`player` image displayed in the GUI.
    """

    disk_used_percent = NumericProperty(0)
    '''Percent of disk usage space in the :attr:`video_recorder` recorder
    directory.
    '''

    def __init__(self, open_player_thread=True, **kwargs):
        super(CeedPlayer, self).__init__(**kwargs)

        self.ffmpeg_player = FFmpegPlayer()
        self.ptgray_player = PTGrayPlayer(open_thread=open_player_thread)
        self.thor_player = ThorCamPlayer(open_thread=open_player_thread)
        self.client_player = RemoteVideoPlayer()

        self.image_file_recorder = ImageFileRecorder()
        self.video_recorder = VideoRecorder()

        self.fbind('player_name', self._update_player)
        self._update_player()

        self.fbind('recorder_name', self._update_recorder)
        self._update_recorder()

        self.ffmpeg_player.display_frame = self.display_frame
        self.ptgray_player.display_frame = self.display_frame
        self.thor_player.display_frame = self.display_frame
        self.client_player.display_frame = self.display_frame

        Clock.schedule_interval(self.update_disk_usage, 0.1)

    def _update_player(self, *largs):
        self.player = getattr(self, '{}_player'.format(self.player_name))
        self.player_settings = getattr(
            self, '{}_settings'.format(self.player_name))

    def _update_recorder(self, *largs):
        self.recorder = getattr(self, '{}_recorder'.format(self.recorder_name))
        self.recorder_settings = getattr(
            self, '{}_recorder_settings'.format(self.recorder_name))

    def create_widgets(self):
        """Creates all the widgets required to show player/recorder.
        """
        self.ffmpeg_settings = FFmpegSettingsWidget(player=self.ffmpeg_player)
        self.ptgray_settings = PTGraySettingsWidget(player=self.ptgray_player)
        self.thor_settings = ThorCamSettingsWidget(player=self.thor_player)
        self.client_settings = ClientPlayerSettingsWidget(
            player=self.client_player)

        self.image_file_recorder_settings = ImageFileRecordSettingsWidget(
            recorder=self.image_file_recorder)
        self.video_recorder_settings = VideoRecordSettingsWidget(
            recorder=self.video_recorder)

    def display_frame(self, image, metadata=None):
        """The displays the image to the user and adds it to :attr:`last_image`.
        """
        app = App.get_running_app()
        widget = app.central_display
        if widget is not None:
            widget.update_img(image)
            self.last_image = image

    def update_disk_usage(self, *largs):
        """Runs periodically to update :attr:`disk_used_percent`.
        """
        p = self.video_recorder.record_directory
        p = 'C:\\' if not exists(p) else (p if isdir(p) else dirname(p))
        if not exists(p):
            p = '/home'
        self.disk_used_percent = round(psutil.disk_usage(p).percent) / 100.

    def load_screenshot(self, app, paths):
        """Loads a previously saved screenshot image file and displayes it.
        """
        if not paths:
            return
        fname = paths[0]
        app.last_directory = dirname(fname)

        if fname.endswith('.h5'):
            img = App.get_running_app().ceed_data.load_last_fluorescent_image(
                fname)
        else:
            images = [m for m in ImageLoader(fname)]

            if not images:
                raise Exception('Could not find image in {}'.format(fname))
            img = images[0][0]

        self.display_frame(img, None)

    def save_screenshot(self, img, app, paths):
        """Saves the currently displayed image to a file.
        """
        if not paths:
            return
        app.last_directory = dirname(paths[0])
        BaseRecorder.save_image(paths[0], img)

    def stop(self):
        """Stops all the players and recorders from playing/recording.
        """
        for player in (
                self.ffmpeg_player, self.thor_player, self.client_player,
                self.image_file_recorder, self.video_recorder,
                self.ptgray_player):
            if player is not None:
                player.stop()

    def clean_up(self):
        """Stops all the players and recorders and cleans up their resources.
        """
        for player in (
                self.ffmpeg_player, self.thor_player, self.client_player,
                self.image_file_recorder, self.video_recorder,
                self.ptgray_player):
            if player is not None:
                player.stop_all(join=True)
