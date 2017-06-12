'''Viewer widgets
=====================

Widgets used on the viewer side of the controller/viewer interface.
These are displayed when the second process of the viewer is running.
'''

from kivy.uix.behaviors.focus import FocusBehavior

from ceed.view.controller import ViewController

__all__ = ('ViewRootFocusBehavior', )


class ViewRootFocusBehavior(FocusBehavior):
    '''Adds focus behavior to the viewer.

    Whenever a key is pressed in the second process it is passed on to the
    controller.
    '''

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        ViewController.send_keyboard_down(keycode[1], modifiers)
        return True

    def keyboard_on_key_up(self, window, keycode):
        ViewController.send_keyboard_up(keycode[1])
        return True
