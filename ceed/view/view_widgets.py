

from kivy.uix.behaviors.focus import FocusBehavior

from ceed.view.controller import ViewController


class ViewRootFocusBehavior(FocusBehavior):

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        ViewController.send_keyboard_down(keycode[1], modifiers)
        return True

    def keyboard_on_key_up(self, window, keycode):
        ViewController.send_keyboard_up(keycode[1])
        return True
