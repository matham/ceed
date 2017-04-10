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

from ceed.utils import WidgetList, ShowMoreSelection, ShowMoreBehavior, \
    fix_name, ColorBackgroundBehavior
from ceed.function import FunctionFactory, FuncGroup


class FuncList(ShowMoreSelection, WidgetList, BoxLayout):

    def add_func(self, name):
        parent = None
        after = None
        if self.selected_nodes:
            widget = self.selected_nodes[0]
            if isinstance(widget, FuncWidgetGroup):
                parent = widget.func
            else:
                after = widget.func
                parent = after.parent_func

        src_func = FunctionFactory.avail_funcs[name]
        if parent:
            if not parent.parent_in_other_children(src_func):
                parent.add_func(deepcopy(src_func), after=after)
        else:
            FunctionFactory.add_func(deepcopy(src_func))

    def get_selectable_nodes(self):
        return list(reversed([
            f.display for func in FunctionFactory.editable_func_list for
            f in func.get_funcs()]))


class FuncWidget(ShowMoreBehavior, BoxLayout):

    func = ObjectProperty(None, rebind=True)

    selected = BooleanProperty(False)

    selection_controller = ObjectProperty(None)

    func_controller = ObjectProperty(None)

    display_parent = ObjectProperty(None)

    def __init__(self, **kwargs):
        kwargs.setdefault('func_controller', FunctionFactory)
        kwargs.setdefault('display_parent', knspace.funcs)
        kwargs.setdefault('selection_controller', knspace.funcs)

        super(FuncWidget, self).__init__(**kwargs)
        self._display_properties()
        self.settings_root.parent.remove_widget(self.settings_root)
        if not isinstance(self, FuncWidgetGroup):
            self.expand.parent.remove_widget(self.expand)

    @property
    def name(self):
        return self.func.name

    def _display_properties(self):
        '''Name is special.
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
        if self.func.parent_func:
            self.func.parent_func.remove_func(self.func)
        else:
            self.func_controller.remove_func(self.func)

    def show_func(self):
        if self.parent:
            return
        parent = self.func.parent_func
        if parent:
            i = len(parent.funcs) - parent.funcs.index(self.func) - 1
            parent.display.more.add_widget(self, index=i)
            self.link_container()
        else:
            self.display_parent.add_widget(self)

    def hide_func(self):
        if self.selected:
            self.display_parent.deselect_node(self)
        elif isinstance(self, FuncWidgetGroup):
            c = self.selected_child()
            if c is not None:
                self.display_parent.deselect_node(c.display)

        if self.parent:
            self.parent.remove_widget(self)


class FuncWidgetGroup(FuncWidget):

    def _show_more(self, *largs):
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
        children = self.func.get_funcs()
        next(children)
        for child in children:
            if child.display.selected:
                return child
        return None


class FuncPropTextWidget(FlatTextInput):

    func = None

    prop_name = ''

    def __init__(self, func=None, prop_name=None, **kwargs):
        super(FuncPropTextWidget, self).__init__(**kwargs)
        self.func = func
        self.prop_name = prop_name
        if not self.hint_text:
            self.hint_text = prop_name
        func.fbind(prop_name, self._update_text)
        self._update_text()

    def _update_text(self, *largs):
        self.text = '{}'.format(getattr(self.func, self.prop_name))

    def _update_attr(self, text):
        if not text:
            self._update_text()
            return

        self.func.track_source = False
        if self.input_filter:
            text = {'int': int, 'float': float}[self.input_filter](text)
        setattr(self.func, self.prop_name, text)


class FuncNamePropTextWidget(FuncPropTextWidget):

    def _update_attr(self, text):
        if not text:
            self._update_text()
            return

        if text != self.func.name:
            self.func.name = fix_name(text, FunctionFactory.avail_funcs)
