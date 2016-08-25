'''Player
==========

Plays and records media.
'''

from os.path import abspath, isdir, dirname, join, exists
import psutil

from cplcom.player import PTGrayPlayer, FFmpegPlayer, VideoMetadata
from cplcom.utils import pretty_space, pretty_time

from kivy.event import EventDispatcher
from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.clock import Clock
from kivy.uix.dropdown import DropDown
from kivy.compat import clock


class CeedPlayer(object):

    def display_frame(self, *largs):
        widget = knspace.central_display
        img = self.last_image
        if widget is not None and img is not None:
            widget.update_img(img[0])


class CeedPTGrayPlayer(CeedPlayer, PTGrayPlayer):
    pass


class CeedFFmpegPlayer(CeedPlayer, FFmpegPlayer):
    pass


class CeedPlayer(KNSpaceBehavior, EventDispatcher):

    player_singleton = None

    pt_player = None

    ff_player = None

    player = None

    pt_player_active = BooleanProperty(False)
    '''True when pt player is playing or being configured.
    '''

    pt_player_play = BooleanProperty(False)
    '''True when pt player is actually playing (excluding when starting or
    stopping to play).
    '''

    player_record_active = BooleanProperty(False)
    '''True when either player is starting, recording, or stopping recording.
    '''

    player_record = BooleanProperty(False)
    '''True when player is actually recordings (excluding when starting or
    stopping to record).
    '''

    last_record_filename = ''

    disk_used_percent = NumericProperty(0)

    play_status = StringProperty('')

    def __init__(self, **kwargs):
        super(CeedPlayer, self).__init__(**kwargs)
        CeedPlayer.player_singleton = self
        self.knsname = 'player'
        self.pt_player = CeedPTGrayPlayer()
        self.ff_player = CeedFFmpegPlayer()
        Clock.schedule_once(self.bind_players)
        Clock.schedule_interval(self.update_cycle, 0.1)

    def update_cycle(self, *largs):
        p = knspace.path_dir.text
        p = 'C:\\' if not exists(p) else (p if isdir(p) else dirname(p))
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
        pt_player = self.pt_player

        def player_active(*largs):
            self.pt_player_active = bool(self.pt_player.play_state != 'none'
                or self.pt_player.config_active)
            self.pt_player_play = self.pt_player.play_state == 'playing'

        pt_player.fbind('play_state', player_active)
        pt_player.fbind('config_active', player_active)

        def pt_config(key, *largs):
            if key == 'ips':
                knspace.pt_ip.values = self.pt_player.ips
            elif key == 'ip':
                knspace.pt_ip.text = self.pt_player.ip
            elif key == 'serials':
                knspace.pt_serial.values = map(str, self.pt_player.serials)
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

        player_active()
        knspace.pt_ip.values = self.pt_player.ips
        knspace.pt_ip.text = self.pt_player.ip
        knspace.pt_serial.values = map(str, self.pt_player.serials)
        knspace.pt_serial.text = '{}'.format(self.pt_player.serial)
        record_paths()
        player_record()
        ff_player_filename()

    def refresh_pt_cams(self):
        self.pt_player.ask_config('serials')

    def reconfig_pt_cams(self):
        self.pt_player.ask_config('gui')

    def set_pt_serial(self, serial):
        self.pt_player.serial = int(serial) if serial else 0

    def set_pt_ip(self, ip):
        self.pt_player.ip = ip

    def set_ff_play_filename(self, filename):
        self.ff_player.play_filename = filename
        self.ff_player.file_fmt = ''

    def update_record_path(self, directory=None, fname=None, count=None):
        pt = self.pt_player
        ff = self.ff_player
        if directory is not None:
            pt.record_directory = ff.record_directory = directory
        if fname is not None:
            pt.record_fname = ff.record_fname = fname
        if count is not None:
            pt.record_fname_count = ff.record_fname_count = count

    def set_filename_widget(self, text_wid, path, selection, filename, is_dir=True):
        if not selection:
            if exists(join(path, filename)):
                selection = [filename]
            else:
                return

        f = abspath(join(path, selection[0]))
        if is_dir and not isdir(f):
            f = dirname(f)
        text_wid.text = f

    def handle_fname(self, fname, count, source='fname'):
        n = count.text
        if source == 'count':
            fname.text = fname.fmt_text = fname.orig_text.replace('{}', n)
        elif not fname.focus:
            fname.orig_text = fname.text
            fname.text = fname.fmt_text = fname.orig_text.replace('{}', n)
        else:
            fname.text = fname.orig_text

    def play(self, live):
        player = self.pt_player if live else self.ff_player
        if self.player or player.play_state != 'none':
            return

        self.last_record_filename = ''
        self.player = player
        player.play()

    def record(self):
        if not self.player:
            return
        if self.player.record_state == 'none':
            self.player.record()
            self.last_record_filename = self.player.record_filename

    def stop(self):
        if not self.player:
            return
        if self.player.play_state in ('starting', 'playing'):
            self.player.stop()

        if self.last_record_filename:
            knspace.record_path = self.last_record_filename
        self.player = None

    def stop_recording(self):
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

    @staticmethod
    def is_player_active():
        player = CeedPlayer.player_singleton
        if not player:
            return False

        return player.pt_player and player.pt_player.play_state != 'none' \
            or player.ff_player and player.ff_player.play_state != 'none'

    def save_config(self):
        pass


class BlankDropDown(DropDown):

    def __init__(self, **kwargs):
        super(BlankDropDown, self).__init__(container=None, **kwargs)
