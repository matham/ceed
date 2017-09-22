import subprocess
import multiprocessing as mp
import traceback
try:
    from pypixxlib import _libdpx as libdpx
    from pypixxlib.propixx import PROPixx
    from pypixxlib.propixx import PROPixxCTRL
except ImportError:
    libdpx = PROPixx = PROPixxCTRL = None


video_modes = ['RGB', 'RB3D', 'RGB240', 'RGB180', 'QUAD4X', 'QUAD12X', 'GREY3X']
led_modes = {'RGB': 0, 'GB': 1, 'RB': 2, 'B': 3, 'RG': 4, 'G': 5, 'R': 6, 'none': 7}


class RemoteTraceback(Exception):

    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb


class ExceptionWithTraceback:

    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb

    def __reduce__(self):
        return rebuild_exc, (self.exc, self.tb)


def rebuild_exc(exc, tb):
    exc.__cause__ = RemoteTraceback(tb)
    return exc


def exec_func(func, queue, *l, **kw):
    exc = None
    import sys
    print(sorted(sys.modules.keys()))
    try:
        func(*l, **kw)
    except Exception as e:
        exc = ExceptionWithTraceback(e, e.__traceback__)
    queue.put(('done', exc))


def run_in_process(func, *largs, **kwargs):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=exec_func, args=(func, queue,) + largs, kwargs=kwargs)
    process.start()
    process.join()

    res, exception = queue.get()
    if res != 'done':
        raise Exception('Something went wrong with the propixx process')

    if exception is not None:
        raise exception


def run_subprocess(name, arg):
    if name == 'pixel':
        arg = 'enablePixelMode' if arg else 'disablePixelMode'
        code = ('from pypixxlib.propixx import PROPixxCTRL; '
                'ctrl = PROPixxCTRL(); '
                'ctrl.dout.{}(); '
                'ctrl.updateRegisterCache(); '
                'ctrl.close()'.format(arg))
    elif name == 'led':
        code = ('from pypixxlib import _libdpx as libdpx; '
                'libdpx.DPxOpen(); '
                'libdpx.DPxSetPPxLedMask({}); '
                'libdpx.DPxUpdateRegCache(); '
                'libdpx.DPxClose()'.format(led_modes[arg]))
    else:
        code = ('from pypixxlib.propixx import PROPixx; '
                'dev = PROPixx(); '
                "dev.setDlpSequencerProgram('{}'); "
                'dev.updateRegisterCache(); '
                'dev.close()'.format(arg))
    subprocess.run('python -c "{}"'.format(code), shell=True, check=True)


def pixel_mode(state):
    if PROPixxCTRL is None:
        raise ImportError('Cannot open PROPixx library')

    ctrl = PROPixxCTRL()
    if state:
        ctrl.dout.enablePixelMode()
    else:
        ctrl.dout.disablePixelMode()
    ctrl.updateRegisterCache()
    ctrl.close()


def led_mode(mode):
    '''Sets the projector's LED mode. ``mode`` can be one of
    :attr:`ViewControllerBase.led_modes`.
    '''
    if libdpx is None:
        raise ImportError('Cannot open PROPixx library')

    libdpx.DPxOpen()
    libdpx.DPxSetPPxLedMask(led_modes[mode])
    libdpx.DPxUpdateRegCache()
    libdpx.DPxClose()


def video_mode(mode):
    '''Sets the projector's video mode. ``mode`` can be one of
    :attr:`ViewControllerBase.video_modes`.
    '''
    if PROPixx is None:
        raise ImportError('Cannot open PROPixx library')

    dev = PROPixx()
    dev.setDlpSequencerProgram(mode)
    dev.updateRegisterCache()
    dev.close()
