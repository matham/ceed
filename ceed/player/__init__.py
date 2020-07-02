"""Video Player
================

Plays and records media from e.g. a camera.
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

    _config_props_ = ('player_name', )

    _config_children_ = {
        'ffmpeg': 'ffmpeg_player', 'ptgray': 'ptgray_player',
        'thor': 'thor_player', 'network_client': 'client_player',
        'image_file_recorder': 'image_file_recorder',
        'video_recorder': 'video_recorder',
    }

    ffmpeg_player: FFmpegPlayer = None

    ffmpeg_settings = None

    ptgray_player: PTGrayPlayer = None

    ptgray_settings = None

    thor_player: ThorCamPlayer = None

    thor_settings = None

    client_player: RemoteVideoPlayer = None

    client_settings = None

    player: BasePlayer = ObjectProperty(None, rebind=True)

    player_settings = ObjectProperty(None)

    player_name = StringProperty('ffmpeg')

    player_to_raw_name_map = {
        'Webcam/File': 'ffmpeg', 'Network': 'client', 'Thor': 'thor',
        'PointGray': 'ptgray'
    }

    player_to_nice_name_map = {v: k for k, v in player_to_raw_name_map.items()}

    image_file_recorder: ImageFileRecorder = None

    image_file_recorder_settings = None

    video_recorder: VideoRecorder = None

    video_recorder_settings = None

    recorder: BaseRecorder = ObjectProperty(None, rebind=True)

    recorder_settings = ObjectProperty(None)

    recorder_name = StringProperty('video')

    recorder_to_raw_name_map = {'Images': 'image_file', 'Video': 'video'}

    recorder_to_nice_name_map = {
        v: k for k, v in recorder_to_raw_name_map.items()}

    last_image = None

    disk_used_percent = NumericProperty(0)
    '''Percent of disk usage space.
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
        """The displays the new image to the user.
        """
        app = App.get_running_app()
        widget = app.central_display
        if widget is not None:
            widget.update_img(image)
            self.last_image = image

    def update_disk_usage(self, *largs):
        """Runs periodically to update the disk usage.
        """
        p = self.video_recorder.record_directory
        p = 'C:\\' if not exists(p) else (p if isdir(p) else dirname(p))
        if not exists(p):
            p = '/home'
        self.disk_used_percent = round(psutil.disk_usage(p).percent) / 100.

    def load_screenshot(self, app, paths):
        """Loads a previously saved screenshot of the camera as a background
        image.
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
        """Saves the image acquired to a file.
        """
        if not paths:
            return
        app.last_directory = dirname(paths[0])
        BaseRecorder.save_image(paths[0], img)

    def stop(self):
        for player in (
                self.ffmpeg_player, self.thor_player, self.client_player,
                self.image_file_recorder, self.video_recorder,
                self.ptgray_player):
            if player is not None:
                player.stop()

    def clean_up(self):
        for player in (
                self.ffmpeg_player, self.thor_player, self.client_player,
                self.image_file_recorder, self.video_recorder,
                self.ptgray_player):
            if player is not None:
                player.stop_all(join=True)
