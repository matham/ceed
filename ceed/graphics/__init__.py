'''Graphcs
==================

Collection of widgets for displaying common things.
'''
from itertools import chain, islice

from kivy.uix.widget import Widget
from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.uix.behaviors.knspace import KNSpaceBehavior
from kivy.uix.behaviors.focus import FocusBehavior
from kivy.event import EventDispatcher
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.utils import get_color_from_hex

__all__ = ('ShowMoreSelection', 'ShowMoreBehavior',
           'TouchSelectBehavior', 'BoxSelector', 'WidgetList')


class ShowMoreSelection(object):

    exapnd_prop = StringProperty('show_more')

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if keycode[1] in ('right', 'left') and self.selected_nodes:
            setattr(
                self.selected_nodes[-1], self.exapnd_prop,
                keycode[1] == 'right')
            return True
        return super(ShowMoreSelection, self).keyboard_on_key_down(
            window, keycode, text, modifiers)


class ShowMoreBehavior(object):

    show_more = BooleanProperty(False)

    more = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(ShowMoreBehavior, self).__init__(**kwargs)
        self.fbind('show_more', self._show_more)
        self._show_more()

    def _show_more(self, *largs):
        if not self.more:
            return

        if self.show_more and self.more not in self.children:
            self.add_widget(self.more)
        elif not self.show_more and self.more in self.children:
            self.remove_widget(self.more)


class TouchSelectBehavior(object):

    controller = ObjectProperty(None)

    use_parent = BooleanProperty(True)

    selectee = ObjectProperty(None)

    def on_touch_up(self, touch):
        if super(TouchSelectBehavior, self).on_touch_up(touch):
            return True
        if touch.grab_current is not None:
            return False
        if self.collide_point(*touch.pos):
            s = self.selectee or (self.parent if self.use_parent else self)
            self.controller.select_with_touch(s, touch)
            return True
        return False


class BoxSelector(TouchSelectBehavior, BoxLayout):
    pass


class WidgetList(KNSpaceBehavior, CompoundSelectionBehavior, FocusBehavior):

    child_name_attr_name = 'name'

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if super(WidgetList, self).keyboard_on_key_down(
            window, keycode, text, modifiers):
            return True
        if self.select_with_key_down(window, keycode, text, modifiers):
            return True
        return False

    def keyboard_on_key_up(self, window, keycode):
        if super(WidgetList, self).keyboard_on_key_up(window, keycode):
            return True
        if self.select_with_key_up(window, keycode):
            return True
        return False

    def select_node(self, node):
        if node in self.selected_nodes:
            return False
        if super(WidgetList, self).select_node(node):
            node.selected = True
            return True
        return False

    def deselect_node(self, node):
        if node not in self.selected_nodes:
            return False
        if super(WidgetList, self).deselect_node(node):
            node.selected = False
            return True
        return False

    def goto_node(self, key, last_node, last_node_idx):
        node, idx = super(WidgetList, self).goto_node(key, last_node,
                                                      last_node_idx)
        if node != last_node:
            return node, idx
        nodes = self.get_selectable_nodes()
        rev = self.nodes_order_reversed
        name = self.child_name_attr_name
        if not last_node:
            last_node_idx = len(nodes) - 1 if rev else 0

        if rev:
            last_node_idx = len(nodes) - 1 - last_node_idx
            names = [n.name for n in reversed(nodes)]
        else:
            names = [n.name for n in nodes]

        try:
            i = names.index(key, last_node_idx + 1)
        except ValueError:
            try:
                i = names.index(key, 0, last_node_idx + 1)
            except ValueError:
                return node, idx

        if rev:
            i = len(nodes) - 1 - i
        return nodes[i], i

Factory.register(classname='ShowMoreSelection', cls=ShowMoreSelection)
Factory.register(classname='ShowMoreBehavior', cls=ShowMoreBehavior)
Factory.register(classname='TouchSelectBehavior', cls=TouchSelectBehavior)
Factory.register(classname='WidgetList', cls=WidgetList)
