'''Viewer widgets
=================

Defines widgets used with the :mod:`~ceed.view` module. These widgets are used
to control and display the experiment on screen, both when playing the
experiment for preview and when playing the experiment full-screen in a second
process.
'''
from time import perf_counter
from typing import List

from kivy.uix.behaviors.focus import FocusBehavior
from kivy.uix.scatter import Scatter
from kivy.properties import NumericProperty, BooleanProperty
from kivy.app import App
from kivy.graphics.vertex_instructions import Point
from kivy.graphics.transformation import Matrix
from kivy.graphics.context_instructions import Color
from kivy.factory import Factory

__all__ = ('ViewRootFocusBehavior', 'MEAArrayAlign')

_get_app = App.get_running_app


class ViewRootFocusBehavior(FocusBehavior):
    """The root widget used for the second process when the experiment is
    played. It adds focus behavior to the viewer.

    Whenever a key is pressed in the second process it is passed on to the
    controller in the main process who handles it as needed (possibly sending
    a message back to the second process).
    """

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        _get_app().view_controller.send_keyboard_down(
            keycode[1], modifiers, perf_counter())
        return True

    def keyboard_on_key_up(self, window, keycode):
        _get_app().view_controller.send_keyboard_up(keycode[1], perf_counter())
        return True


class MEAArrayAlign(Scatter):
    """The widget used during the experiment design to help align the MEA
    electrode array to the camera and projector.

    It displays a grid of points that you can align to the real-world camera
    acquired picture of the electrode grid. See :mod:`~ceed.view.controller`
    for more details.
    """

    num_rows = NumericProperty(12)
    """Number of rows.

    See :attr:`~ceed.view.controller.ViewControllerBase.mea_num_rows`
    """

    num_cols = NumericProperty(12)
    """Number of columns.

    See :attr:`~ceed.view.controller.ViewControllerBase.mea_num_cols`
    """

    pitch = NumericProperty(20)
    """The distance in pixels between the rows/columns.

    See :attr:`~ceed.view.controller.ViewControllerBase.mea_pitch`
    """

    diameter = NumericProperty(3)
    """Diameter of each electrode circle in pixels..

    See :attr:`~ceed.view.controller.ViewControllerBase.mea_diameter`
    """

    show = BooleanProperty(False)
    """Whether the grid is currently shown.
    """

    color = None
    """The grid color.
    """

    label = None
    """The label that shows the "A1" corner electrode.
    """

    label2 = None
    """The label that shows the "M1" corner electrode.
    """

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
        """Automatic callback that updates the graphics whenever any parameter
        changes.
        """
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
    def make_matrix(elems: List[List[float]]) -> Matrix:
        """Converts a matrix represented as a 2D list to a kivy Matrix.
        """
        mat = Matrix()
        mat.set(array=elems)
        return mat

    @staticmethod
    def compare_mat(mat: Matrix, mat_list: List[List[float]]) -> bool:
        """Compares a matrix represented as a 2D list to a kivy Matrix object
        and returns whether they are equivalent.
        """
        return mat.tolist() == tuple(tuple(item) for item in mat_list)
