import os
from pytest_kivy.app import AsyncUnitApp

from ceed.main import CeedApp

__all__ = ('CeedTestGUIApp', 'CeedTestApp')


class CeedTestGUIApp(CeedApp):

    def __init__(self, ini_file, **kwargs):
        self._ini_config_filename = ini_file
        self._data_path = os.path.dirname(ini_file)
        super().__init__(**kwargs)

    def check_close(self):
        super().check_close()
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


class CeedTestApp(AsyncUnitApp):

    app: CeedTestGUIApp
