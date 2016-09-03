from itertools import chain, islice

from kivy.uix.widget import Widget
from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.uix.behaviors.knspace import KNSpaceBehavior
from kivy.uix.behaviors.focus import FocusBehavior
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.clock import Clock
from kivy.factory import Factory


class ExpandWidget(Widget):

    is_open = BooleanProperty(False)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.ud['expand_used'] = True
            return True
        return super(ExpandWidget, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.ud.get('expand_used', False):
            return True
        return super(ExpandWidget, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.ud.get('expand_used', False) and \
                self.collide_point(*touch.pos):
            self.is_open = not self.is_open
            return True
        return super(ExpandWidget, self).on_touch_up(touch)


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
            return True
        return False

    def deselect_node(self, node):
        if node not in self.selected_nodes:
            return False
        if super(WidgetList, self).deselect_node(node):
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
            names = [getattr(n, name) for n in reversed(nodes)]
        else:
            names = [getattr(n, name) for n in nodes]

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

Factory.register(classname='WidgetList', cls=WidgetList)
