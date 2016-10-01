from collections import defaultdict
from copy import deepcopy
import re

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, ListProperty, DictProperty
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.compat import string_types

from ceed.utils import WidgetList, ShowMoreSelection, ShowMoreBehavior, \
    fix_name
from ceed.function import FunctionFactory, CeedFunc, FuncGroup


class FuncList(ShowMoreSelection, WidgetList, BoxLayout):

    funcs = DictProperty({})

    editable_func_list = []

    def __init__(self, **kwargs):
        super(FuncList, self).__init__(**kwargs)
        self.editable_func_list = []
        funcs = self.funcs
        for cls in FunctionFactory.get_classes():
            f = cls()
            funcs[f.name] = f

    def save_funcs(self, id_map=None):
        if id_map is None:
            id_map = {}
        CeedFunc.get_id_map(self.editable_func_list, id_map)

        states = [f._copy_state() for f in self.editable_func_list]
        return states, id_map

    def recover_funcs(self, funcs, id_map):
        funcs = [CeedFunc.recover_func(state) for state in funcs]
        CeedFunc.set_source_from_id(funcs, id_map)
        for f in funcs:
            self._add_func(f)
        return funcs

    def add_func(self, name):
        parent = None
        if self.selected_nodes:
            widget = self.selected_nodes[0]
            if isinstance(widget, FuncWidgetGroup):
                parent = widget.func
            else:
                parent = widget.func.parent_func

        src_func = self.funcs[name]
        if parent:
            if not parent.parent_in_other_children(src_func):
                func = deepcopy(src_func)
                parent.add_func(func)
                parent.display.add_widget_func(func)
        else:
            self._add_func(deepcopy(src_func))

    def _add_func(self, func):
        func.source_func = None
        func.name = fix_name(func.name, self.funcs)

        self.add_widget(func.display)
        assert not func.source_func
        func.fbind('name', self._track_func_name, func)
        self.funcs[func.name] = func
        self.editable_func_list.append(func)

    def remove_func(self, func):
        if func.display.selected:
            self.deselect_node(func.display)
        elif isinstance(func.display, FuncWidgetGroup):
            c = func.display.selected_child()
            if c is not None:
                self.deselect_node(c.display)
        self.remove_widget(func.display)
        func.funbind('name', self._track_func_name, func)
        del self.funcs[func.name]
        self.editable_func_list.remove(func)

    def _track_func_name(self, func, *largs):
        for name, f in self.funcs.items():
            if f is func:
                if func.name == name:
                    return

                del self.funcs[name]
                break
        func.name = fix_name(func.name, self.funcs)
        self.funcs[func.name] = func

    def get_selectable_nodes(self):
        return list(reversed([f.display for func in self.editable_func_list for
                              f in func.get_children_funcs()]))


class FuncWidget(ShowMoreBehavior, BoxLayout):

    func = ObjectProperty(None, rebind=True)

    selected = BooleanProperty(False)

    def __init__(self, func, **kwargs):
        super(FuncWidget, self).__init__(**kwargs)
        self.func = func
        self._display_properties()

    @property
    def name(self):
        return self.func.name

    def _display_properties(self):
        '''Name is special.
        '''
        func = self.func
        items = func.get_gui_elements()
        kwargs = func.get_gui_controls()
        add = self.more.add_widget

        input_types = {'int': 'int', 'float': 'float', int: 'int',
                        float: 'float', 'str': 'str', str: 'str'}
        input_filter = {'float': 'float', 'int': 'int', 'str': None}
        props = defaultdict(list)

        assert 'name' in kwargs
        if not func.source_func:
            add(FuncNamePropTextWidget(func=func, prop_name='name'))
            s = self.ids['source_control']
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
            label = Factory.XSizedLabel
            for fmt, keys in sorted(props.items(), key=lambda x: x[0]):
                for key in sorted(keys):
                    grid.add_widget(label(text=key, padding_x='10dp'))
                    grid.add_widget(FuncPropTextWidget(
                        func=func, prop_name=key,
                        input_filter=input_filter[fmt]))
            add(grid)

        for item in items:
            add(item)

    def remove_from_parent(self):
        if self.func.parent_func:
            self.func.parent_func.remove_func(self.func)
        else:
            knspace.funcs.remove_func(self.func)


class FuncWidgetGroup(FuncWidget):

    func_container = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(FuncWidgetGroup, self).__init__(**kwargs)
        self.remove_widget(self.func_container)
        add = self.add_widget_func
        for func in self.func.funcs:
            add(func)

    def _show_more(self, *largs):
        super(FuncWidgetGroup, self)._show_more()
        if self.show_more:
            self.add_widget(self.func_container)
        else:
            c = self.selected_child()
            if c is not None:
                knspace.funcs.deselect_node(c.display)
            self.remove_widget(self.func_container)

    def add_widget_func(self, func):
        self.func_container.add_widget(func.display)

    def remove_widget_func(self, widget):
        if widget.selected:
            knspace.funcs.deselect_node(widget)
        elif isinstance(widge, FuncWidgetGroup):
            c = widget.selected_child()
            if c is not None:
                knspace.funcs.deselect_node(c.display)
        self.func_container.remove_widget(widget)

    def selected_child(self):
        children = self.func.get_children_funcs()
        next(children)
        for child in children:
            if child.display.selected:
                return child
        return None


class FuncPropTextWidget(TextInput):

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
            self.func.name = fix_name(text, knspace.funcs.funcs)
