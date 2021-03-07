# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import os
import pathlib
from kivy_deps import sdl2, glew
import nixio
import ffpyplayer
import base_kivy_app
import cpl_media
import ceed
from kivy.tools.packaging.pyinstaller_hooks import get_deps_minimal, \
    get_deps_all, hookspath, runtime_hooks

kwargs = get_deps_minimal(video=None, audio=None, camera=None)
kwargs['hiddenimports'].extend([
    'ffpyplayer', 'ffpyplayer.pic', 'win32timezone',
    'ffpyplayer.threading', 'ffpyplayer.tools', 'ffpyplayer.writer',
    'ffpyplayer.player', 'ffpyplayer.player.clock', 'ffpyplayer.player.core',
    'ffpyplayer.player.decoder', 'ffpyplayer.player.frame_queue',
    'ffpyplayer.player.player', 'ffpyplayer.player.queue',
    'numpy.random.common', 'numpy.random.bounded_integers',
    'numpy.random.entropy', 'plyer.platforms.win.filechooser',
    'plyer.facades.filechooser'])

data_files = []

root = pathlib.Path(nixio.__file__).parent
f = root.joinpath('info.json')
data_files.append((str(f), str(f.relative_to(root.parent).parent)))


root = pathlib.Path(ceed.__file__).parent
for pat in ('**/plugin/**/*.py', '**/plugin/*.py'):
    for f in root.glob(pat):
        data_files.append((str(f), str(f.relative_to(root.parent).parent)))

a = Analysis(['../ceed/run_app.py'],
             pathex=['.'],
             datas=base_kivy_app.get_pyinstaller_datas() + cpl_media.get_pyinstaller_datas() + ceed.get_pyinstaller_datas() + data_files,
             hookspath=hookspath(),
             runtime_hooks=runtime_hooks(),
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=os.environ.get('CEED_NOARCHIVE', '0') == '1',
             **kwargs)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins + ffpyplayer.dep_bins)],
          name='Ceed',
          debug=os.environ.get('CEED_NOARCHIVE', '0') == '1',
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          icon='..\\doc\\source\\media\\ceed_icon.ico')
