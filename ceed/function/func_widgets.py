'''Function Widgets
=======================

Defines the GUI components used with :mod:`ceed.function`.
'''
from collections import defaultdict
from copy import deepcopy

from kivy.uix.boxlayout import BoxLayout
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, ListProperty, DictProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.compat import string_types
from kivy.app import App
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivy.lang import Builder

from cplcom.graphics import FlatTextInput
from kivy_garden.drag_n_drop import DraggableLayoutBehavior

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

    func_controller = ObjectProperty(None)
    '''The controller to which the function is added or removed from.
    This is e.g. :attr:`ceed.function.FunctionFactoryBase` in the function list
    case or the stage to which the function is attached.
    '''

    is_visible = BooleanProperty(False)

    _selector = None

    _settings = None

    theme_interpolation = 0

    settings_root = None

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
        noise_supported_parameters = func.get_noise_supported_parameters()
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
            grid = Factory.XYSizedGridLayout(cols=3)
            label = Factory.FlatXSizedLabel
            color = App.get_running_app().theme.text_primary
            for fmt, keys in sorted(props.items(), key=lambda x: x[0]):
                for key in sorted(keys):
                    grid.add_widget(
                        label(text=pretty_names.get(key, key),
                              padding_x='10dp', flat_color=color))

                    widget = FuncPropTextWidget(input_filter=input_filter[fmt])
                    widget.func = func
                    widget.prop_name = key
                    widget.apply_binding()
                    grid.add_widget(widget)

                    if key in noise_supported_parameters:
                        noise = Factory.NoiseSelection()
                        noise.func = func
                        noise.prop_name = key
                        grid.add_widget(noise)
                    else:
                        grid.add_widget(
                            Widget(size_hint=(None, None), size=(0, 0)))

            for key, cls in sorted(cls_widgets, key=lambda x: x[0]):
                cls, kw = cls
                if isinstance(cls, string_types):
                    cls = Factory.get(cls)

                grid.add_widget(
                    label(text=pretty_names.get(key, key),
                          padding_x='10dp', flat_color=color))
                grid.add_widget(cls(
                    func=func, prop_name=key, **kw))

                if key in noise_supported_parameters:
                    noise = Factory.NoiseSelection()
                    noise.more_widget = FuncNoiseDropDown()
                    grid.add_widget(noise)
                else:
                    grid.add_widget(
                        Widget(size_hint=(None, None), size=(0, 0)))

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
        self.selection_controller = selection_controller

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

    def apply_kv(self):
        func = self.ref_func or self.func
        app = _get_app()
        func.fbind('on_changed', app.changed_callback)

        if self.ref_func is None:
            self.settings_root = settings_root = FuncSettingsDropDown(
                func_widget=self)
            self._settings = settings_root.settings

        Builder.apply_rules(self, 'FuncWidgetStyle', dispatch_kv_post=True)


class FuncWidgetGroup(FuncWidget):
    '''The widget associated with :class:`ceed.function.FuncGroup`.
    '''

    children_container = None

    theme_interpolation = 0.25

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

    def apply_kv(self):
        super(FuncWidgetGroup, self).apply_kv()
        if self.ref_func is not None:
            return

        self.more = self.children_container = func_list = GroupFuncList()
        self.add_widget(func_list)
        func_list.group_widget = self

        Builder.apply_rules(
            func_list, 'GroupFuncListStyle', dispatch_kv_post=True)

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

    _binding = None

    def apply_binding(self):
        if not self.hint_text:
            self.hint_text = self.prop_name
        uid = self.func.fbind(self.prop_name, self._update_text)
        self._binding = self.func, self.prop_name, uid
        self._update_text()

    def unbind_tracking(self):
        if self._binding is None:
            return
        obj, prop, uid = self._binding
        obj.unbind_uid(prop, uid)
        self._binding = None

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
            setattr(self.func, self.prop_name, self.text)

    def _update_values(self, *largs):
        vals = self.values_getter()

        if self.allow_empty:
            vals.insert(0, '')
        self.values = vals

        if self.text not in vals:
            self.text = vals[0] if vals else ''


class FuncSettingsDropDown(Factory.FlatDropDown):

    func_widget = ObjectProperty(None)

    def __init__(self, func_widget, **kwargs):
        self.func_widget = func_widget
        super(FuncSettingsDropDown, self).__init__(**kwargs)


class FuncNoiseDropDown(Factory.FlatDropDown):

    noise_param = ObjectProperty(None, allownone=True, rebind=True)

    noise_factory = ObjectProperty(None, rebind=True)

    func = None

    prop_name = ''

    param_container = None

    def initialize_param(self, func, prop_name):
        self.clear_noise_param()
        self.func = func
        self.prop_name = prop_name
        self.noise_factory = func.function_factory.param_noise_factory
        self.noise_param = func.noisy_parameters.get(prop_name, None)

    def clear_noise_param(self):
        for widget in self.param_container.children[::2]:
            widget.unbind_tracking()
        self.param_container.clear_widgets()

        self.noise_param = None

    def set_noise_instance(self, cls_name):
        if self.prop_name in self.func.noisy_parameters:
            del self.func.noisy_parameters[self.prop_name]

        if cls_name == 'none':
            return

        self.noise_param = self.noise_factory.get_cls(cls_name)()

    def show_noise_params(self):
        noise_param = self.noise_param
        if noise_param is None:
            return

        label = Factory.FlatXSizedLabel
        color = App.get_running_app().theme.text_primary
        grid = self.param_container
        pretty_names = noise_param.get_prop_pretty_name()

        for prop, val in sorted(
                noise_param.get_config().items(), key=lambda x: x[0]):
            if prop == 'cls':
                continue

            grid.add_widget(
                label(text=pretty_names.get(prop, prop),
                      padding_x='10dp', flat_color=color))

            widget = FuncPropTextWidget(input_filter='float')
            widget.func = noise_param
            widget.prop_name = prop
            widget.apply_binding()
            grid.add_widget(widget)
