'''Viewer widgets
=====================

Widgets used on the viewer side of the controller/viewer interface.
These are displayed when the second process of the viewer is running.
'''
from math import isclose

from kivy.uix.behaviors.focus import FocusBehavior
from kivy.uix.stencilview import StencilView
from kivy.uix.scatter import Scatter
from kivy.clock import Clock
from kivy.properties import NumericProperty, BooleanProperty
from kivy.app import App
from kivy.graphics.vertex_instructions import Point
from kivy.graphics.transformation import Matrix
from kivy.graphics.context_instructions import Color
from kivy.factory import Factory

__all__ = ('ViewRootFocusBehavior', 'MEAArrayAlign')

_get_app = App.get_running_app


class ViewRootFocusBehavior(FocusBehavior):
    '''Adds focus behavior to the viewer.

    Whenever a key is pressed in the second process it is passed on to the
    controller.
    '''

    _ctrl_down = False

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if keycode[1] in ('ctrl', 'lctrl', 'rctrl'):
            self._ctrl_down = True
        _get_app().view_controller.send_keyboard_down(keycode[1], modifiers)
        return True

    def keyboard_on_key_up(self, window, keycode):
        if keycode[1] in ('ctrl', 'lctrl', 'rctrl'):
            self._ctrl_down = False

        if self._ctrl_down:
            if keycode[1] == 'q':
                _get_app().view_controller.filter_background = \
                    not _get_app().view_controller.filter_background
                return True
        _get_app().view_controller.send_keyboard_up(keycode[1])
        return True

    def keyboard_on_textinput(self, window, text):
        if not self._ctrl_down:
            return True

        if text == '+':
            _get_app().view_controller.alpha_color = min(
                1., _get_app().view_controller.alpha_color + .01)
        elif text == '-':
            _get_app().view_controller.alpha_color = max(
                0., _get_app().view_controller.alpha_color - .01)
        return True


class MEAArrayAlign(Scatter):

    num_rows = NumericProperty(12)

    num_cols = NumericProperty(12)

    pitch = NumericProperty(20)

    diameter = NumericProperty(3)

    show = BooleanProperty(False)

    color = None

    label = None

    label2 = None

    def __init__(self, **kwargs):
        super(MEAArrayAlign, self).__init__(**kwargs)
        label = self.label = Factory.XYSizedLabel(text='A1')
        self.add_widget(label)
        label2 = self.label2 = Factory.XYSizedLabel(text='M1')
        self.add_widget(label2)
        self.fbind('num_rows', self.update_graphics)
        self.fbind('num_cols', self.update_graphics)
        self.fbind('pitch', self.update_graphics)
        self.fbind('diameter', self.update_graphics)
        self.update_graphics()

        def track_show(*largs):
            label.color = 1, 1, 1, (1 if self.show else 0)
            label2.color = 1, 1, 1, (1 if self.show else 0)
        self.fbind('show', track_show)
        track_show()

    def update_graphics(self, *largs):
        self.canvas.remove_group('MEAArrayAlign')
        pitch = self.pitch
        radius = self.diameter / 2.0

        with self.canvas:
            self.color = Color(
                1, 1, 1, 1 if self.show else 0, group='MEAArrayAlign')
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    Point(
                        points=[col * pitch, row * pitch], pointsize=radius,
                        group='MEAArrayAlign')

        h = max((self.num_rows - 1) * pitch, 0)
        w = max((self.num_cols - 1) * pitch, 0)
        self.label.y = h
        self.label2.y = 0
        self.label2.right = self.label.right = w
        self.size = w, h + 35

    def on_touch_down(self, touch):
        if not self.do_translation_x and \
                not self.do_translation_y and \
                not self.do_rotation and \
                not self.do_scale:
            return False

        return super(MEAArrayAlign, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if not self.do_translation_x and \
                not self.do_translation_y and \
                not self.do_rotation and \
                not self.do_scale:
            return False

        return super(MEAArrayAlign, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if not self.do_translation_x and \
                not self.do_translation_y and \
                not self.do_rotation and \
                not self.do_scale:
            return False

        return super(MEAArrayAlign, self).on_touch_up(touch)

    @staticmethod
    def make_matrix(elems):
        mat = Matrix()
        mat.set(array=elems)
        return mat

    @staticmethod
    def compare_mat(mat, mat_list):
        return mat.tolist() == tuple(tuple(item) for item in mat_list)
