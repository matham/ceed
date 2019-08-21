'''Player
==========

Plays and records media from e.g. a camera.

Various player objects can play different filet types. For example
:class:`CeedPTGrayPlayer` will play video from point gray cameras while
:class:`CeedFFmpegPlayer` plays videos that ffmpeg can play.
'''

from os.path import abspath, isdir, dirname, join, exists
import psutil
from ffpyplayer.pic import ImageLoader

from cplcom.player import PTGrayPlayer, FFmpegPlayer, VideoMetadata, Player
from cplcom.utils import pretty_space, pretty_time

from kivy.event import EventDispatcher
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.clock import Clock
from kivy.uix.dropdown import DropDown
from kivy.compat import clock
from kivy.app import App

__all__ = ('CeedPlayerBase', 'CeedPTGrayPlayer', 'CeedFFmpegPlayer',
           'CeedPlayer')


class CeedPlayerBase(object):
    '''The underlying player class used in all player types.
    '''

    def display_frame(self, *largs):
        '''The displays the last image to the user.
        '''
        app = App.get_running_app()
        widget = app.central_display
        img = self.last_image
        if widget is not None and img is not None:
            widget.update_img(img[0])
            app.player.last_image = img[0]
            if app.use_remote_view:
                App.get_running_app().remote_viewer.send_image(img[0])


class CeedPTGrayPlayer(CeedPlayerBase, PTGrayPlayer):
    '''Plays and records point gray media.
    '''
    pass


class CeedFFmpegPlayer(CeedPlayerBase, FFmpegPlayer):
    '''Plays and records ffmpeg media.
    '''
    pass


class CeedRemotePlayer(CeedPlayerBase):

    last_image = None

    display_trigger = None

    def __init__(self, **kwargs):
        super(CeedRemotePlayer, self).__init__(**kwargs)
        self.display_trigger = Clock.create_trigger(self.display_frame, 0)

    def display_frame(self, *largs):
        '''The displays the last image to the user.
        '''
        app = App.get_running_app()
        widget = app.central_display
        img = self.last_image
        if widget is not None and img is not None:
            widget.update_img(img)
            app.player.last_image = img


class CeedPlayer(EventDispatcher):
    '''Controls the media player/recorder in ceed.

    A singlton instance of this class controls both a :class:`CeedPTGrayPlayer`
    and a :class:`CeedFFmpegPlayer` instance and selects the media source to
    play and record based on which of them is currently in :attr:`player`.
    '''

    __settings_attrs__ = ('browse_path', )

    player_singleton = None
    '''The singleton instance of this class.
    '''

    pt_player = None
    '''A :class:`CeedPTGrayPlayer` instance.
    '''

    ff_player = None
    '''A :class:`CeedFFmpegPlayer` instance.
    '''

    player = None
    '''Either :attr:`pt_player` or :attr:`ff_player` and is the player
    currently used.
    '''

    pt_player_active = BooleanProperty(False)
    '''True when :attr:`pt_player` is playing or being configured.
    '''

    pt_player_play = BooleanProperty(False)
    '''True when :attr:`pt_player` is actually playing (excluding when starting
    or stopping to play).
    '''

    ff_player_play = BooleanProperty(False)
    '''True when :attr:`CeedFFmpegPlayer` is actually playing (excluding when
    starting or stopping to play).
    '''

    player_record_active = BooleanProperty(False)
    '''True when either :attr:`pt_player` or :attr:`ff_player` is starting,
    recording, or stopping recording.
    '''

    player_record = BooleanProperty(False)
    '''True when :attr:`player` is actually recordings (excluding when
    starting or stopping to record).
    '''

    browse_path = StringProperty('')
    '''Path where the filebrowser opens to.
    '''

    last_record_filename = ''
    '''The full path to the last recorded video file.
    '''

    disk_used_percent = NumericProperty(0)
    '''Percent of disk usage space.
    '''

    play_status = StringProperty('')
    '''A string showing some :attr:`player` statistics e.g. frame rate etc.
    '''

    last_image = None
    '''The last image instance that was gotten from the :attr:`player`.
    '''

    _pt_settings_last = ''

    _pt_settings_remote_last = ''

    def __init__(self, **kwargs):
        super(CeedPlayer, self).__init__(**kwargs)
        CeedPlayer.player_singleton = self
        self.knsname = 'player'
        pt = self.pt_player = CeedPTGrayPlayer()
        pt.estimate_record_rate = True
        self.ff_player = CeedFFmpegPlayer()
        # Clock.schedule_once(self.bind_players)
        # Clock.schedule_interval(self.update_cycle, 0.1)

    def update_cycle(self, *largs):
        '''Runs periodically to update the status and statistics.
        '''
        p = knspace.path_dir.text
        p = 'C:\\' if not exists(p) else (p if isdir(p) else dirname(p))
        if not exists(p):
            p = '/home'
        self.disk_used_percent = round(psutil.disk_usage(p).percent) / 100.

        player = self.player
        if not player:
            return
        if player.record_state == 'recording':
            elapsed = pretty_time(max(0, clock() - player.ts_record))
            size = pretty_space(player.size_recorded)
            self.play_status = (
                '{}s ({}) [color=#FF0000]{}[/color] {}fps'.
                format(elapsed, size, player.frames_skipped,
                       int(player.real_rate)))
        else:
            s = self.play_status.rsplit(' ', 1)[0].strip()
            self.play_status = '{} {}fps'.format(s, int(player.real_rate))

    def bind_players(self, *largs):
        '''Connects all the ffmpeg and point gray instance properties such
        that when they update, the GUI is also updated.
        '''
        pt_player = self.pt_player

        def player_active(*largs):
            self.pt_player_active = bool(self.pt_player.play_state != 'none'
                or self.pt_player.config_active)
            self.pt_player_play = self.pt_player.play_state == 'playing'

            settings = self.get_valid_pt_settings()
            knspace.pt_settings_opt.values = settings

            if knspace.gui_remote_view.state == 'down':
                App.get_running_app().remote_viewer.send_cam_settings(
                    'cam_settings', settings)

        pt_player.fbind('play_state', player_active)
        pt_player.fbind('config_active', player_active)

        def player_active_ff(*largs):
            self.ff_player_play = self.ff_player.play_state == 'playing'

        self.ff_player.fbind('play_state', player_active_ff)

        def pt_config(key, *largs):
            if key == 'ips':
                knspace.pt_ip.values = self.pt_player.ips
            elif key == 'ip':
                knspace.pt_ip.text = self.pt_player.ip
            elif key == 'serials':
                knspace.pt_serial.values = list(
                    map(str, self.pt_player.serials))
            elif key == 'serial':
                knspace.pt_serial.text = '{}'.format(self.pt_player.serial)

        pt_player.fbind('ips', pt_config, 'ips')
        pt_player.fbind('ip', pt_config, 'ip')
        pt_player.fbind('serials', pt_config, 'serials')
        pt_player.fbind('serial', pt_config, 'serial')

        def record_paths(*largs):
            knspace.path_dir.text = self.pt_player.record_directory
            t = knspace.path_fname.orig_text = knspace.path_fname.text = \
                self.pt_player.record_fname
            n = knspace.path_count.text = self.pt_player.record_fname_count
            knspace.path_fname.text = knspace.path_fname.fmt_text = \
                t.replace('{}', n)

        pt_player.fbind('record_directory', record_paths)
        pt_player.fbind('record_fname', record_paths)
        pt_player.fbind('record_fname_count', record_paths)

        def player_record(*largs):
            self.player_record_active = self.pt_player.record_state != 'none' \
                or self.ff_player.record_state != 'none'
            self.player_record = self.pt_player.record_state == 'recording' \
                or self.ff_player.record_state == 'recording'

        pt_player.fbind('record_state', player_record)
        self.ff_player.fbind('record_state', player_record)

        def ff_player_filename(*largs):
            knspace.record_path.text = self.ff_player.play_filename
        self.ff_player.fbind('play_filename', ff_player_filename)

        def track_state(*largs):
            if not self.player:
                return
            if self.player.record_state in ('starting', 'recording'):
                knspace.gui_record.state = 'down'
            else:
                knspace.gui_record.state = 'normal'

            if not self.player:
                return
            if self.player.play_state in ('starting', 'playing'):
                knspace.gui_play.state = 'down'
            else:
                knspace.gui_play.state = 'normal'

        pt_player.fbind('play_state', track_state)
        pt_player.fbind('record_state', track_state)
        self.ff_player.fbind('play_state', track_state)
        self.ff_player.fbind('record_state', track_state)
        self.bind_pt_setting(knspace.pt_settings_opt.text)

        player_active()
        knspace.pt_ip.values = self.pt_player.ips
        knspace.pt_ip.text = self.pt_player.ip
        knspace.pt_serial.values = list(map(str, self.pt_player.serials))
        knspace.pt_serial.text = '{}'.format(self.pt_player.serial)
        record_paths()
        player_record()
        ff_player_filename()

    def _update_pt_setting(self, *largs):
        opts = getattr(self.pt_player, self._pt_settings_last)
        knspace.gui_pt_settings_opt_auto.state = 'down' if opts['auto'] else 'normal'
        knspace.gui_pt_settings_opt_min.text = '{:0.2f}'.format(opts['min'])
        knspace.gui_pt_settings_opt_max.text = '{:0.2f}'.format(opts['max'])
        knspace.gui_pt_settings_opt_value.text = '{:0.2f}'.format(opts['value'])
        knspace.gui_pt_settings_opt_disable.state = 'normal' if opts['controllable'] else 'down'
        knspace.gui_pt_settings_opt_slider.min = opts['min']
        knspace.gui_pt_settings_opt_slider.max = opts['max']
        knspace.gui_pt_settings_opt_slider.value = opts['value']

    def bind_pt_setting(self, setting):
        if self._pt_settings_last:
            self.pt_player.funbind(self._pt_settings_last, self._update_pt_setting)
        self._pt_settings_last = ''

        if setting:
            self._pt_settings_last = setting
            self.pt_player.fbind(setting, self._update_pt_setting)
            self._update_pt_setting()

    def _update_pt_setting_remote(self, *largs):
        if knspace.gui_remote_view.state == 'down':
            App.get_running_app().remote_viewer.send_cam_settings(
                'cam_setting',
                (self._pt_settings_remote_last,
                 dict(getattr(self.pt_player, self._pt_settings_remote_last))
                 )
            )

    def bind_pt_remote_setting(self, setting):
        if self._pt_settings_remote_last:
            self.pt_player.funbind(
                self._pt_settings_remote_last, self._update_pt_setting_remote)
        self._pt_settings_remote_last = ''

        if setting:
            self._pt_settings_remote_last = setting
            self.pt_player.fbind(setting, self._update_pt_setting_remote)
            self._update_pt_setting_remote()

    def get_valid_pt_settings(self):
        settings = []
        for setting in [
                'brightness', 'exposure', 'sharpness', 'hue', 'saturation',
                'gamma', 'shutter', 'gain', 'iris', 'frame_rate', 'pan',
                'tilt']:
            opts = getattr(self.pt_player, setting)
            if opts.get('present', False):
                settings.append(setting)
        return list(sorted(settings))

    def refresh_pt_cams(self):
        '''Causes the point gray cams to refresh.
        '''
        self.pt_player.ask_config('serials')

    def change_pt_setting_opt(self, setting, name, value):
        self.pt_player.ask_cam_option_config(setting, name, value)

    def reload_pt_setting_opt(self, setting):
        self.pt_player.ask_cam_option_config(setting, '', None)

    def reconfig_pt_cams(self):
        '''Shows the point gray config GUI.
        '''
        self.pt_player.ask_config('gui')

    def set_pt_serial(self, serial):
        '''Selects the point gray camera to use by serial number.
        '''
        self.pt_player.serial = int(serial) if serial else 0

    def set_pt_ip(self, ip):
        '''Selects the point gray camera to use by IP number.
        '''
        self.pt_player.ip = ip

    def set_ff_play_filename(self, filename):
        '''Selects the file that the ffmpeh player will play.
        '''
        self.ff_player.play_filename = filename
        self.ff_player.file_fmt = ''

    def update_record_path(self, directory=None, fname=None, count=None):
        '''Called by the GUI to set the recording filename.
        '''
        pt = self.pt_player
        ff = self.ff_player
        if directory is not None:
            pt.record_directory = ff.record_directory = directory
        if fname is not None:
            pt.record_fname = ff.record_fname = fname
        if count is not None:
            pt.record_fname_count = ff.record_fname_count = count

    def set_filename_widget(self, text_wid, path, selection, filename,
                            is_dir=True):
        '''Called by the GUI to set the filename to the ``text_wid``
        TextInput and to update the default browsing path.
        '''
        if not selection:
            if exists(join(path, filename)):
                selection = [filename]
            else:
                return

        f = abspath(join(path, selection[0]))
        if is_dir and not isdir(f):
            f = dirname(f)
        text_wid.text = f
        self.browse_path = path

    def load_screenshot(self, path, selection, filename):
        '''Loads a previously saved screenshot of the camera as a background
        image.
        '''
        if not isdir(path) or not filename:
            raise Exception('Invalid path or filename')

        self.browse_path = path
        fname = join(path, filename)

        if filename.endswith('.h5'):
            img = App.get_running_app().ceed_data.load_last_fluorescent_image(
                fname)
        else:
            images = [m for m in ImageLoader(fname)]

            if not images:
                raise Exception('Could not find image in {}'.format(fname))
            img = images[0][0]

        App.get_running_app().central_display.update_img(img)
        self.last_image = img

    def set_screenshot(self, image):
        App.get_running_app().central_display.update_img(image)
        self.last_image = image

    def save_screenshot(self, img, path, selection, filename):
        '''Saves the image acquired to a file.
        '''
        if not isdir(path) or not filename:
            raise Exception('Invalid path or filename')
        self.browse_path = path
        fname = join(path, filename)

        if exists(fname):
            def yesno_callback(overwrite):
                if not overwrite:
                    return
                Player.save_image(fname, img)

            yesno = App.get_running_app().yesno_prompt
            yesno.msg = ('"{}" already exists, would you like to '
                         'overwrite it?'.format(fname))
            yesno.callback = yesno_callback
            yesno.open()
        else:
            Player.save_image(fname, img)

    def handle_fname(self, fname, count, source='fname'):
        '''Properly formats the filename of the recording video displayed
        in the GUI.
        '''
        n = count.text
        if source == 'count':
            fname.text = fname.fmt_text = fname.orig_text.replace('{}', n)
        elif not fname.focus:
            fname.orig_text = fname.text
            fname.text = fname.fmt_text = fname.orig_text.replace('{}', n)
        else:
            fname.text = fname.orig_text

    def play(self, live):
        '''Starts playing the :attr:`player`.
        '''
        player = self.pt_player if live else self.ff_player
        if self.player or player.play_state != 'none':
            return

        self.last_record_filename = ''
        self.player = player
        player.play()

    def record(self):
        '''Starts recording from the :attr:`player`.
        '''
        if not self.player:
            return
        if self.player.record_state == 'none':
            self.player.record()
            self.last_record_filename = self.player.record_filename

    def set_pause(self, state):
        '''Sets the :attr:`player` to pause, but only for the ffmpeg player.
        The point gray cam doesn't support pause.
        '''
        if self.player is self.ff_player and \
                self.player.play_state == 'playing':
            self.player.play_paused = state

    def stop(self):
        '''Stops playing the :attr:`player`.
        '''
        if not self.player:
            return
        if self.player.play_state in ('starting', 'playing'):
            self.player.stop()

        if self.last_record_filename:
            knspace.record_path.text = self.last_record_filename
        self.player = None

    def stop_recording(self):
        '''Stops recording from the :attr:`player`.
        '''
        if not self.player or self.player.record_state == 'none':
            return

        if self.player.record_state != 'stopping':
            self.player.stop_recording()
        if '{}' in self.player.record_fname:
            self.player.record_fname_count = \
                str(int(self.player.record_fname_count) + 1)

    @staticmethod
    def exit_players():
        '''Closes all running players and saves the last config.
        '''
        player = CeedPlayer.player_singleton
        if player:
            for p in (player.pt_player, player.ff_player):
                if not p:
                    continue
                try:
                    p.stop_all(join=True)
                except:
                    pass
            player.save_config()
            CeedPlayer.player_singleton = None

    @staticmethod
    def is_player_active():
        '''Returns True if any of the players are currently playing.
        '''
        player = CeedPlayer.player_singleton
        if not player:
            return False

        return player.pt_player and player.pt_player.play_state != 'none' \
            or player.ff_player and player.ff_player.play_state != 'none'

    def save_config(self):
        '''Saves the config to a file (not really).
        '''
        pass
