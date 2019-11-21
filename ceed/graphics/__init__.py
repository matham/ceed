'''Graphcs
==================

Collection of widgets for displaying common things.
'''
from itertools import chain, islice

from kivy.uix.widget import Widget
from kivy.uix.behaviors.compoundselection import CompoundSelectionBehavior
from kivy.uix.behaviors.focus import FocusBehavior
from kivy.event import EventDispatcher
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.uix.slider import Slider
from kivy.utils import get_color_from_hex

from kivy_garden.drag_n_drop import DraggableController, DraggableObjectBehavior

__all__ = ('ShowMoreSelection', 'ShowMoreBehavior',
           'TouchSelectBehavior', 'BoxSelector', 'WidgetList',
           'CeedDragNDrop', 'CeedDraggableObjectBehavior',
           'FilterTouchEagerlyBehavior')


class ShowMoreSelection(object):
    '''This class is meant to be co-inherited from with a
    :class:`kivy.uix.behaviors.compoundselection.CompoundSelectionBehavior`.

    When a right or left keyboard key is pressed while a child of the resulting
    widget is selected (i.e. a node is selected) this class sets the
    :attr:`exapnd_prop` property of the selected widget to True or False
    for right and left respectively.
    '''

    exapnd_prop = StringProperty('show_more')
    '''The name of the property of the selected widget that will be set
    to True or False. See class description.
    '''

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if keycode[1] in ('right', 'left') and self.selected_nodes:
            setattr(
                self.selected_nodes[-1], self.exapnd_prop,
                keycode[1] == 'right')
            return True
        return super(ShowMoreSelection, self).keyboard_on_key_down(
            window, keycode, text, modifiers)


class ShowMoreBehavior(object):
    '''Behavior that displays or hides the :attr:`more` widget when
    :attr:`show_more` is set to True or False respectively.
    '''

    show_more = BooleanProperty(False)
    '''Whether the :attr:`more` widget is displayed as a child of this
    instance or removed.
    '''

    more = ObjectProperty(None)
    '''The widget to display as a child of the instance when :attr:`show_more`
    is True.
    '''

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
    '''Behavior meant to be used as the child of a
    :class:`kivy.uix.behaviors.compoundselection.CompoundSelectionBehavior`
    and adds touch selection to the child.

    Specifically, when the child is touched this will select/de-select this
    instance using the :attr:`controller`.
    '''

    controller = ObjectProperty(None)
    '''The
    :class:`kivy.uix.behaviors.compoundselection.CompoundSelectionBehavior`
    instance through which this instance is selected.
    '''

    use_parent = BooleanProperty(True)
    '''Whether the parent of this widget should be selected upon touch (True)
    or whether the widget in :attr:`selectee` should be selected (False)
    upon the touch.
    '''

    selectee = ObjectProperty(None)
    '''When :attr:`use_parent` is False, the widget stored in :attr:`selectee`
    is selected rather than the parent of this widget.
    '''

    def on_touch_up(self, touch):
        if super(TouchSelectBehavior, self).on_touch_up(touch):
            return True
        if touch.grab_current is not None:
            return False
        if self.collide_point(*touch.pos) and self.collide_point(*touch.opos):
            s = self.selectee or (self.parent if self.use_parent else self)
            self.controller.select_with_touch(s, touch)
            return True
        return False


class BoxSelector(TouchSelectBehavior, BoxLayout):
    '''Combines the touch selection with a box layout.
    '''
    pass


class WidgetList(CompoundSelectionBehavior, FocusBehavior):
    '''A
    :class:`kivy.uix.behaviors.compoundselection.CompoundSelectionBehavior`
    based class with some convenience methods.

    Mainly, when a keyboard key is typed it'll try to select the widget
    whose text property called :attr:`child_name_attr_name` starts with that
    string.
    '''

    child_name_attr_name = 'name'
    '''The propery name to use when sorting or searching for text associated
    with the widgets to be selected. This is the property used to jump to a
    widget when its "name" is typed.
    '''

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


class CeedDragNDrop(DraggableController):
    '''Adds ``KNSpaceBehavior`` to the drag controller.
    '''
    pass


class CeedDraggableObjectBehavior(DraggableObjectBehavior):
    '''Adds the ``on_drag_init`` event, which is dispatched when
    ``initiate_drag`` is called.
    '''

    __events__ = ('on_drag_init', )

    drag_copy = BooleanProperty(True)
    '''Whether when the drag started the function was copied or was removed
    from its parent and should be moved.
    '''

    obj_dragged = ObjectProperty(None)
    '''The higher level object associated with the drag.
    '''

    def initiate_drag(self):
        super(CeedDraggableObjectBehavior, self).initiate_drag()
        self.dispatch('on_drag_init')

    def on_drag_init(self, *largs):
        pass


class FilterTouchEagerlyBehavior(object):

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        return super(FilterTouchEagerlyBehavior, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            print('filtered', touch)
            return False
        print('passed', touch)
        return super(FilterTouchEagerlyBehavior, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        return super(FilterTouchEagerlyBehavior, self).on_touch_up(touch)


Factory.register(classname='ShowMoreSelection', cls=ShowMoreSelection)
Factory.register(classname='ShowMoreBehavior', cls=ShowMoreBehavior)
Factory.register(classname='TouchSelectBehavior', cls=TouchSelectBehavior)
Factory.register(classname='WidgetList', cls=WidgetList)
Factory.register(classname='CeedDragNDrop', cls=CeedDragNDrop)
Factory.register(classname='CeedDraggableObjectBehavior',
                 cls=CeedDraggableObjectBehavior)
Factory.register(classname='FilterTouchEagerlyBehavior',
                 cls=FilterTouchEagerlyBehavior)
