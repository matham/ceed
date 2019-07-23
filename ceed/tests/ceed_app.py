import os

from kivy.config import Config
Config.set('graphics', 'width', '1600')
Config.set('graphics', 'height', '900')
for items in Config.items('input'):
    Config.remove_option('input', items[0])


from ceed.main import CeedApp
from kivy.tests.async_common import UnitKivyApp

kv_loaded = False


class CeedTestApp(CeedApp, UnitKivyApp):

    def __init__(self, ini_file, **kwargs):
        self._ini_config_filename = ini_file
        self._data_path = os.path.dirname(ini_file)
        super(CeedTestApp, self).__init__(**kwargs)

    def load_app_kv(self):
        global kv_loaded
        if kv_loaded:
            return

        super(CeedTestApp, self).load_app_kv()
        kv_loaded = True

    def check_close(self):
        super(CeedTestApp, self).check_close()
        return True

    def handle_exception(self, msg, exc_info=None,
                         level='error', *largs):
        super(CeedApp, self).handle_exception(
            msg, exc_info, level, *largs)

        if isinstance(exc_info, str):
            self.get_logger().error(msg)
            self.get_logger().error(exc_info)
        elif exc_info is not None:
            tp, value, tb = exc_info
            try:
                if value is None:
                    value = tp()
                if value.__traceback__ is not tb:
                    raise value.with_traceback(tb)
                raise value
            finally:
                value = None
                tb = None
        elif level in ('error', 'exception'):
            raise Exception(msg)
