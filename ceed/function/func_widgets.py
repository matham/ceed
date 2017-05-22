'''Function Widgets
=======================

Defines the GUI components used with :mod:`ceed.function`.
'''
from collections import defaultdict
from copy import deepcopy
import re

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, ListProperty, DictProperty
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.compat import string_types
from kivy.app import App

from cplcom.graphics import FlatTextInput

from ceed.utils import fix_name
from ceed.graphics import WidgetList, ShowMoreSelection, ShowMoreBehavior, \
    ColorBackgroundBehavior
from ceed.function import FunctionFactory, FuncGroup

__all__ = ('FuncList', 'FuncWidget', 'FuncWidgetGroup', 'FuncPropTextWidget',
           'FuncNamePropTextWidget')


class FuncList(ShowMoreSelection, WidgetList, BoxLayout):
    '''Widgets that shows the list of available functions and allows for the
    creation of new functions.
    '''

    def add_func(self, name):
        '''Adds a copy of the the function with the given ``name`` to the
        available functions or to a function group.
        '''
        parent = None
        after = None
        if self.selected_nodes:
            widget = self.selected_nodes[0]
            if isinstance(widget, FuncWidgetGroup):
                parent = widget.func
            else:
                after = widget.func
                parent = after.parent_func

        src_func = FunctionFactory.funcs_inst[name]
        if parent:
            if not parent.parent_in_other_children(src_func):
                parent.add_func(deepcopy(src_func), after=after)
        else:
            FunctionFactory.add_func(deepcopy(src_func))

    def get_selectable_nodes(self):
        return list(reversed([
            f.display for func in FunctionFactory.funcs_user for
            f in func.get_funcs()]))


class FuncWidget(ShowMoreBehavior, BoxLayout):
    '''The widget associated with :class:`ceed.function.CeedFunc`.

    It contains all the configuration options of the function.

    The class is reused anywhere a function is shown in the GUI, including
    in stages.
    '''

    func = ObjectProperty(None, rebind=True)
    '''The :class:`ceed.function.BaseFunc` instance associated with this
    widget.
    '''

    selected = BooleanProperty(False)
    '''Whether the function is selected in the GUI.
    '''

    selection_controller = ObjectProperty(None)
    '''The container that gets called to select the widget when the user
    selects it with a touch. E.g. :class:`FuncList` in the function listing
    case.
    '''

    func_controller = ObjectProperty(None)
    '''The controller to which the function is added or removed from.
    This is e.g. :attr:`ceed.function.FunctionFactory` in the function list
    case or the stage to which the function is attached.
    '''

    display_parent = ObjectProperty(None)
    '''The widget container to which the widget is added or removed when
    displayed.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('func_controller', FunctionFactory)
        kwargs.setdefault('display_parent', knspace.funcs)
        kwargs.setdefault('selection_controller', knspace.funcs)

        super(FuncWidget, self).__init__(**kwargs)
        self.display_properties()
        self.settings_root.parent.remove_widget(self.settings_root)
        if not isinstance(self, FuncWidgetGroup):
            self.expand.parent.remove_widget(self.expand)

    @property
    def name(self):
        '''The :attr:`ceed.function.FuncBase.name` of the function.
        '''
        return self.func.name

    def display_properties(self):
        '''Constructs the configuration option widgets for the function using
        :meth:`ceed.function.FuncBase.get_gui_elements` and
        :meth:`ceed.function.FuncBase.get_gui_props`.
        '''
        func = self.func
        items = func.get_gui_elements()
        kwargs = func.get_gui_props()
        add = self.settings.add_widget

        input_types = {'int': 'int', 'float': 'float', int: 'int',
                        float: 'float', 'str': 'str', str: 'str'}
        input_filter = {'float': 'float', 'int': 'int', 'str': None}
        props = defaultdict(list)

        assert 'name' in kwargs
        if not func.source_func:
            w = FuncNamePropTextWidget(func=func, prop_name='name')
            add(w)
            if func.parent_func:
                w.disabled = True
            s = self.ids.source_control
            s.parent.remove_widget(s)
        del kwargs['name']

        for key, value in kwargs.items():
            if value is not None:
                if value in input_types:
                    props[input_types[value]].append(key)
                else:
                    raise TypeError('"{}" is not a recognized type'.
                                    format(value))
            else:
                value = getattr(func, key)
                if isinstance(value, int):
                    props['int'].append(key)
                elif isinstance(value, float):
                    props['float'].append(key)
                elif isinstance(value, string_types):
                    props['str'].append(key)
                else:
                    raise TypeError('"{}" is not a recognized type'.
                                    format(value))

        if props:
            grid = Factory.XYSizedGridLayout(cols=2)
            label = Factory.FlatXSizedLabel
            color = App.get_running_app().theme.text_primary
            for fmt, keys in sorted(props.items(), key=lambda x: x[0]):
                for key in sorted(keys):
                    grid.add_widget(label(text=key, padding_x='10dp', flat_color=color))
                    grid.add_widget(FuncPropTextWidget(
                        func=func, prop_name=key,
                        input_filter=input_filter[fmt]))
            add(grid)

        for item in items:
            add(item)

    def link_container(self):
        '''Fills in the values of :attr:`selection_controller`,
        :attr:`func_controller`, and :attr:`display_parent` of all the
        functions and sub-functions of this function.
        '''
        parent = self.func.parent_func
        if not parent:
            return
        if parent.display.func_controller is self.func_controller:
            return
        func_controller = parent.display.func_controller
        display_parent = parent.display.display_parent
        selection_controller = parent.display.selection_controller
        for func in self.func.get_funcs():
            func.display.func_controller = func_controller
            func.display.display_parent = display_parent
            func.display.selection_controller = selection_controller

    def remove_from_parent(self):
        '''Removes the function from the its parent or controller if
        it has no parent.
        '''
        if self.func.parent_func:
            self.func.parent_func.remove_func(self.func)
        else:
            self.func_controller.remove_func(self.func)

    def show_func(self, after_index=None):
        '''Shows the function's widget in its widget container.
        '''
        if self.parent:
            return
        parent = self.func.parent_func
        if parent:
            i = len(parent.funcs) - parent.funcs.index(self.func) - 1
            parent.display.more.add_widget(self, index=i)
            self.link_container()
        elif after_index is not None:
            i = len(self.display_parent.children) - after_index - 1
            self.display_parent.add_widget(self, index=i)
        else:
            self.display_parent.add_widget(self)

    def hide_func(self):
        '''Removes the function's widget from its widget container.
        '''
        if self.selected:
            self.display_parent.deselect_node(self)
        elif isinstance(self, FuncWidgetGroup):
            c = self.selected_child()
            if c is not None:
                self.display_parent.deselect_node(c.display)

        if self.parent:
            self.parent.remove_widget(self)


class FuncWidgetGroup(FuncWidget):
    '''The widget associated with :class:`ceed.function.FuncGroup`.
    '''

    def _show_more(self, *largs):
        '''Displays the additional configuration options in the GUI.
        '''
        super(FuncWidgetGroup, self)._show_more()
        if not self.show_more:
            c = self.selected_child()
            if c is not None:
                self.display_parent.deselect_node(c.display)

    def show_func(self):
        super(FuncWidgetGroup, self).show_func()
        for f in self.func.funcs:
            f.display.show_func()

    def selected_child(self):
        '''Returns the child or sub-child etc. :class:`ceed.function.FuncBase`
        that is selected in the GUI or None.
        '''
        children = self.func.get_funcs()
        next(children)
        for child in children:
            if child.display.selected:
                return child
        return None


class FuncPropTextWidget(FlatTextInput):
    '''The widget used to edit a specific configuration option of a
    :class:`ceed.function.FuncBase`.
    '''

    func = None
    '''The :class:`ceed.function.FuncBase` instance it's associated with.
    '''

    prop_name = ''
    '''The name of the property of :attr:`func` that this widget edits.
    '''

    def __init__(self, func=None, prop_name=None, **kwargs):
        super(FuncPropTextWidget, self).__init__(**kwargs)
        self.func = func
        self.prop_name = prop_name
        if not self.hint_text:
            self.hint_text = prop_name
        func.fbind(prop_name, self._update_text)
        self._update_text()

    def _update_text(self, *largs):
        '''Updates the GUI from the function.
        '''
        self.text = '{}'.format(getattr(self.func, self.prop_name))

    def _update_attr(self, text):
        '''Updates the function property from the GUI.
        '''
        if not text:
            self._update_text()
            return

        self.func.track_source = False
        if self.input_filter:
            text = {'int': int, 'float': float}[self.input_filter](text)
        setattr(self.func, self.prop_name, text)


class FuncNamePropTextWidget(FuncPropTextWidget):
    '''The widget used to edit the :attr:`ceed.function.FuncBase.name` of a
    :class:`ceed.function.FuncBase`.
    '''

    def _update_attr(self, text):
        if not text:
            self._update_text()
            return

        if text != self.func.name:
            self.func.name = fix_name(text, FunctionFactory.funcs_inst)
