'''Function Widgets
=======================

Defines the GUI components used with :mod:`ceed.function`.
'''
from collections import defaultdict
from copy import deepcopy

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, ListProperty, DictProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.compat import string_types
from kivy.app import App
from kivy.lang.compiler import kv, KvContext, KvRule
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp

from cplcom.graphics import FlatTextInput
from cplcom.drag_n_drop import DraggableLayoutBehavior

from ceed.utils import fix_name
from ceed.graphics import WidgetList, ShowMoreSelection, ShowMoreBehavior, \
    BoxSelector
from ceed.function import CeedFuncRef, CeedFunc, FuncGroup

__all__ = ('FuncList', 'FuncWidget', 'FuncWidgetGroup', 'FuncPropTextWidget',
           'FuncNamePropTextWidget')

_get_app = App.get_running_app


class FuncList(DraggableLayoutBehavior, ShowMoreSelection, WidgetList,
               BoxLayout):
    '''Widgets that shows the list of available functions and allows for the
    creation of new functions.
    '''

    function_factory = None

    is_visible = BooleanProperty(True)

    def handle_drag_release(self, index, drag_widget):
        if drag_widget.drag_cls == 'func_spinner':
            func = self.function_factory.funcs_inst[
                drag_widget.obj_dragged.text]
        else:
            dragged = drag_widget.obj_dragged
            func = dragged.ref_func or dragged.func

        func = deepcopy(func)
        self.function_factory.add_func(func)

        widget = FuncWidget.get_display_cls(func)()
        widget.initialize_display(func, self.function_factory, self)
        self.add_widget(widget)

    def add_func(self, name):
        '''Adds a copy of the the function with the given ``name`` to the
        available functions or to a function group.
        '''
        parent = None
        after = None
        if self.selected_nodes:
            widget = self.selected_nodes[-1]
            if isinstance(widget, FuncWidgetGroup):
                parent = widget.func
            else:
                after = widget.func
                parent = after.parent_func

        src_func = self.function_factory.funcs_inst[name]

        if parent:
            if parent.can_other_func_be_added(src_func):
                func = self.function_factory.get_func_ref(func=src_func)
                parent.add_func(func, after=after)

                widget = FuncWidget.get_display_cls(func)()
                widget.initialize_display(func, self.function_factory, self)
                parent.display.add_widget(widget)
        else:
            func = deepcopy(src_func)
            self.function_factory.add_func(func)

            widget = FuncWidget.get_display_cls(func)()
            widget.initialize_display(func, self.function_factory, self)
            self.add_widget(widget)

    def get_selectable_nodes(self):
        # a ref func will never be in the root list, so get_funcs will not be
        # called on it
        return list(reversed([
            f.display for func in self.function_factory.funcs_user for
            f in func.get_funcs(step_into_ref=False) if f.display.is_visible]))

    def clear_all(self):
        for widget in self.children[:]:
            self.remove_widget(widget)

    def show_function(self, func):
        widget = FuncWidget.get_display_cls(func)()
        widget.initialize_display(func, self.function_factory, self)
        self.add_widget(widget)


class GroupFuncList(DraggableLayoutBehavior, BoxLayout):

    is_visible = BooleanProperty(False)

    group_widget = None
    """The function's widget containing this list. """

    def handle_drag_release(self, index, drag_widget):
        group_widget = self.group_widget
        group_func = group_widget.func
        function_factory = group_func.function_factory

        if drag_widget.drag_cls == 'func_spinner' or \
                drag_widget.drag_cls == 'func' and \
                drag_widget.obj_dragged.func_controller is function_factory \
                and drag_widget.obj_dragged.func.parent_func is None:
            if drag_widget.drag_cls == 'func_spinner':
                func = function_factory.get_func_ref(
                    name=drag_widget.obj_dragged.text)
            else:
                assert not isinstance(
                    drag_widget.obj_dragged.func, CeedFuncRef)
                func = function_factory.get_func_ref(
                    func=drag_widget.obj_dragged.func)
            if not group_func.can_other_func_be_added(func):
                function_factory.return_func_ref(func)
                return

            group_func.add_func(func, index=len(self.children) - index)

            widget = FuncWidget.get_display_cls(func)()
            widget.initialize_display(
                func, function_factory, group_widget.selection_controller)
            self.add_widget(widget, index=index)
        else:
            dragged = drag_widget.obj_dragged
            if not group_widget.func.can_other_func_be_added(dragged.func):
                return

            func = deepcopy(dragged.func)
            group_widget.func.add_func(func, index=len(self.children) - index)

            widget = FuncWidget.get_display_cls(func)()
            widget.initialize_display(
                func, function_factory, group_widget.selection_controller)
            self.add_widget(widget, index=index)


class FuncWidget(ShowMoreBehavior, BoxLayout):
    '''The widget associated with :class:`ceed.function.CeedFunc`.

    It contains all the configuration options of the function.

    The class is reused anywhere a function is shown in the GUI, including
    in stages.
    '''

    func = None
    '''The :class:`ceed.function.BaseFunc` instance associated with this
    widget.
    '''

    ref_func = None
    '''Func being referenced, if func is the ref. '''

    selected = BooleanProperty(False)
    '''Whether the function is selected in the GUI.
    '''

    selection_controller = None
    '''The container that gets called to select the widget when the user
    selects it with a touch. E.g. :class:`FuncList` in the function listing
    case.
    '''

    func_controller = None
    '''The controller to which the function is added or removed from.
    This is e.g. :attr:`ceed.function.FunctionFactoryBase` in the function list
    case or the stage to which the function is attached.
    '''

    is_visible = BooleanProperty(False)

    _selector = None

    _settings = None

    @property
    def name(self):
        if self.ref_func:
            return self.ref_func.name
        return self.func.name

    @staticmethod
    def get_display_cls(func):
        if isinstance(func, FuncGroup):
            return FuncWidgetGroup
        return FuncWidget

    def display_properties(self):
        '''Constructs the configuration option widgets for the function using
        :meth:`ceed.function.FuncBase.get_gui_elements` and
        :meth:`ceed.function.FuncBase.get_gui_props`.
        '''
        func = self.func
        items = func.get_gui_elements()
        kwargs = func.get_gui_props()
        pretty_names = func.get_prop_pretty_name()
        add = self._settings.add_widget

        input_types = {'int': 'int', 'float': 'float', int: 'int',
                        float: 'float', 'str': 'str', str: 'str'}
        input_filter = {'float': 'float', 'int': 'int', 'str': None}
        props = defaultdict(list)
        cls_widgets = []

        assert 'name' in kwargs
        del kwargs['name']

        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, (list, tuple)):
                    cls_widgets.append((key, value))
                elif value in input_types:
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

        if props or cls_widgets:
            grid = Factory.XYSizedGridLayout(cols=2)
            label = Factory.FlatXSizedLabel
            color = App.get_running_app().theme.text_primary
            for fmt, keys in sorted(props.items(), key=lambda x: x[0]):
                for key in sorted(keys):
                    grid.add_widget(
                        label(text=pretty_names.get(key, key),
                              padding_x='10dp', flat_color=color))
                    grid.add_widget(FuncPropTextWidget(
                        func=func, prop_name=key,
                        input_filter=input_filter[fmt]))

            for key, cls in sorted(cls_widgets, key=lambda x: x[0]):
                cls, kw = cls
                if isinstance(cls, string_types):
                    cls = Factory.get(cls)

                grid.add_widget(
                    label(text=pretty_names.get(key, key),
                          padding_x='10dp', flat_color=color))
                grid.add_widget(cls(
                    func=func, prop_name=key, **kw))

            add(grid)

        for item in items:
            if isinstance(item, string_types):
                item = Factory.get(item)()
            add(item)

    def initialize_display(
            self, func, func_controller, selection_controller):
        '''Fills in the values of :attr:`selection_controller`,
        :attr:`func_controller` of all the
        functions and sub-functions of this function.
        '''
        if isinstance(func, CeedFuncRef):
            self.ref_func = func.func
        func.display = self
        self.func = func

        self.func_controller = func_controller  # or _get_app().function_factory
        self.selection_controller = selection_controller  # or knspace.funcs

        self.apply_kv()
        if self.ref_func is None:
            self.display_properties()

    def remove_func(self):
        if self.func.parent_func:
            self.func.parent_func.remove_func(self.func)
            self.parent.remove_widget(self)
        else:
            if self.func_controller.remove_func(self.func):
                self.parent.remove_widget(self)

        if self.ref_func is not None:
            self.func.function_factory.return_func_ref(self.func)
        else:
            for func in self.func.get_funcs(step_into_ref=False):
                if isinstance(func, CeedFuncRef):
                    func.function_factory.return_func_ref(func)

        if self.selected:
            self.selection_controller.deselect_node(self)

    def handle_expand_widget(self, expand):
        expand.parent.remove_widget(expand)

    def replace_ref_func_with_source(self):
        assert self.ref_func is not None
        if self.func.parent_func is not None:
            controller = self.func.parent_func
        else:
            controller = self.func_controller

        parent_widget = self.parent
        parent_widget.remove_widget(self)
        func, i = controller.replace_ref_func_with_source(self.func)

        widget = FuncWidget.get_display_cls(func)()
        widget.initialize_display(
            func, self.func_controller, self.selection_controller)
        parent_widget.add_widget(
            widget, index=len(parent_widget.children) - i)

        self.func.function_factory.return_func_ref(self.func)

    @kv(proxy='app*')
    def apply_kv(self):
        self.size_hint_y = None
        self.orientation = 'vertical'
        app = _get_app()
        func = self.ref_func or self.func
        func.fbind('on_changed', app.changed_callback)
        interpolation = .25 if isinstance(self.func, FuncGroup) else 0

        with KvContext():
            self.height @= self.minimum_height
            self.size_hint_min_x @= self.minimum_width
            self.is_visible @= self.parent is not None and self.parent.is_visible

            with BoxSelector(
                    parent=self, size_hint_y=None, height='34dp',
                    orientation='horizontal', spacing='5dp', padding='5dp'
                    ) as selector:
                selector.size_hint_min_x @= selector.minimum_width
                selector.controller @= self.selection_controller

                with selector.canvas:
                    color = Color()
                    color.rgba ^= app.theme.interpolate(
                        app.theme.primary_light, app.theme.primary, interpolation) if \
                        not self.selected else app.theme.primary
                    rect = Rectangle()
                    rect.size ^= selector.size
                    rect.pos ^= selector.pos

                with Factory.DraggingWidget(
                        parent=selector, drag_widget=selector,
                        obj_dragged=self, drag_cls='func') as dragger:
                    dragger.drag_copy = True  # root.func.parent_func is None

                with Factory.ExpandWidget(parent=selector) as expand:
                    expand.state = 'down'
                    self.show_more @= expand.is_open
                self.handle_expand_widget(expand)

                with Factory.FlatLabel(
                        parent=selector, center_texture=False,
                        padding=('5dp', '5dp')) as func_label:
                    func_label.flat_color = app.theme.text_primary

                    func_label.text @= func.name
                    func_label.size_hint_min_x @= func_label.texture_size[0]

                with Factory.FlatImageButton(
                        parent=selector, scale_down_color=True,
                        source='flat_delete.png') as del_btn:
                    del_btn.flat_color @= app.theme.accent
                    del_btn.disabled @= func.has_ref if self.ref_func is None else False
                    with KvRule(del_btn.on_release, triggered_only=True):
                        self.remove_func()

                if self.ref_func:
                    with KvContext():
                        with Factory.FlatImageButton(
                                parent=selector, scale_down_color=True,
                                source='call-split.png') as split_btn:
                            split_btn.flat_color @= app.theme.accent

                            with KvRule(split_btn.on_release, triggered_only=True):
                                self.replace_ref_func_with_source()
                else:
                    with KvContext():
                        with Factory.FlatImageButton(
                                parent=selector, scale_down_color=True,
                                source='flat_dots_vertical.png') as more_btn:
                            more_btn.flat_color @= app.theme.accent

                            settings_root, splitter = self.get_settings_dropdown()
                            with KvRule(more_btn.on_release, triggered_only=True):
                                assert self.ref_func is None
                                settings_root.open(selector)
                                splitter.width = max(selector.width, splitter.width)

    @kv(proxy='app*')
    def get_settings_dropdown(self):
        assert self.ref_func is None
        app = _get_app()
        with KvContext():
            with Factory.FlatDropDown(
                    do_scroll=(False, False)) as settings_root:
                settings_root.flat_color @= app.theme.primary_text
                settings_root.flat_border_color @= app.theme.divider

                with Factory.FlatSplitter(
                    parent=settings_root, size_hint=(None, None),
                        sizable_from='left') as splitter:
                    splitter.flat_color @= app.theme.accent
                    splitter.height @= splitter.minimum_height
                    splitter.min_size @= splitter.minimum_width

                    with BoxLayout(
                        parent=splitter, size_hint_y=None, orientation='vertical',
                            spacing='5dp', padding='5dp') as settings:
                        self._settings = settings
                        settings.height @= settings.minimum_height
                        settings.size_hint_min_x @= settings.minimum_width

                        with FuncNamePropTextWidget(
                            parent=settings, func=self.ref_func or self.func,
                                prop_name='name') as name:
                            pass
                        if self.func.parent_func is not None:
                            name.disabled = True

                        with Factory.FlatLabel(
                            parent=settings, padding=('5dp', '5dp'),
                                size_hint_y=None, halign='center') as desc:
                            desc.flat_color @= app.theme.text_primary
                            desc.height @= desc.texture_size[1]
                            desc.text_size @= desc.width, None
                            desc.text @= self.func.description

        return settings_root, splitter


class FuncWidgetGroup(FuncWidget):
    '''The widget associated with :class:`ceed.function.FuncGroup`.
    '''

    children_container = None

    def remove_func(self):
        c = self.selected_child()
        if c is not None:
            self.selection_controller.deselect_node(c.display)

        super(FuncWidgetGroup, self).remove_func()

    def initialize_display(self, func, func_controller, selection_controller):
        super(FuncWidgetGroup, self).initialize_display(
            func, func_controller, selection_controller)

        if self.ref_func:
            return

        for child in func.funcs:
            display = FuncWidget.get_display_cls(child)()
            self.children_container.add_widget(display)
            display.initialize_display(
                child, func_controller, selection_controller)

    @kv(proxy='app*')
    def apply_kv(self):
        FuncWidget.apply_kv(self)
        if self.ref_func:
            return

        app = _get_app()
        with KvContext():
            with GroupFuncList(
                    parent=self, spacing='5dp', size_hint_y=None,
                    orientation='vertical') as more:
                self.more = self.children_container = more
                more.group_widget = self
                more.is_visible @= self.show_more and self.is_visible

                with KvRule(more.children):
                    if more.children:
                        more.padding = '5dp', '5dp', 0, 0
                    else:
                        more.padding = '5dp', '5dp', 0, '5dp'

                more.height @= more.minimum_height
                more.size_hint_min_x @= more.minimum_width

                more.spacer_props = {
                    'size_hint_y': None, 'height': '50dp',
                    'size_hint_min_x': '40dp'}
                more.drag_classes = ['func', 'func_spinner']
                more.controller @= self.func
                with more.canvas:
                    color = Color()
                    color.rgba ^= app.theme.divider

                    more_rect = Rectangle()
                    more_rect.pos ^= more.x + dp(1), more.y
                    more_rect.size ^= dp(2), more.height - dp(5)

    def _show_more(self, *largs):
        '''Displays the additional configuration options in the GUI.
        '''
        super(FuncWidgetGroup, self)._show_more()
        if not self.show_more:
            c = self.selected_child()
            if c is not None:
                self.selection_controller.deselect_node(c.display)

    def selected_child(self):
        '''Returns the child or sub-child etc. :class:`ceed.function.FuncBase`
        that is selected in the GUI or None.
        '''
        if self.func is None:  # XXX: hack because _show_more calls this
            return None
        children = self.func.get_funcs(step_into_ref=False)
        next(children)
        for child in children:
            if child.display.selected:
                return child
        return None

    def handle_expand_widget(self, expand):
        pass


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
            self.func.name = fix_name(
                text, _get_app().function_factory.funcs_inst)


class TrackOptionsSpinner(Factory.SizedCeedFlatSpinner):

    func = None
    '''The :class:`ceed.function.FuncBase` instance it's associated with.
    '''

    prop_name = ''
    '''The name of the property of :attr:`func` that this widget edits.
    '''

    allow_empty = False

    track_obj = None

    track_prop = ''

    values_getter = lambda x: x

    update_items_on_press = BooleanProperty(False)

    _value_trigger = None

    def __init__(
            self, func=None, prop_name=None, allow_empty=False, track_obj=None,
            track_prop='', values_getter=lambda x: x, **kwargs):
        super(TrackOptionsSpinner, self).__init__(**kwargs)
        self.func = func
        self.prop_name = prop_name
        self.allow_empty = allow_empty
        self.track_obj = track_obj
        self.track_prop = track_prop
        self.values_getter = values_getter
        self._value_trigger = Clock.create_trigger(self._update_values, -1)

        if self.update_items_on_press:
            self.spinner.fbind('on_press', self._value_trigger)
        else:
            track_obj.fbind(track_prop, self._value_trigger)
        func.fbind(prop_name, self._update_text)
        self.fbind('text', self._update_attr)
        self._update_text()
        self._update_values()

    def _update_text(self, *largs):
        '''Updates the GUI from the function.
        '''
        self.text = getattr(self.func, self.prop_name)

    def _update_attr(self, *largs):
        '''Updates the function property from the GUI.
        '''
        if getattr(self.func, self.prop_name) != self.text:
            self.func.track_source = False
            setattr(self.func, self.prop_name, self.text)

    def _update_values(self, *largs):
        vals = self.values_getter()

        if self.allow_empty:
            vals.insert(0, '')
        self.values = vals

        if self.text not in vals:
            self.text = vals[0] if vals else ''
