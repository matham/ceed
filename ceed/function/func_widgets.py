"""Function Widgets
=======================

Defines the GUI components used with :mod:`ceed.function`.
"""
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Union, Type, TypeVar

from kivy.uix.boxlayout import BoxLayout
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, ListProperty, DictProperty
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.compat import string_types
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder

from base_kivy_app.graphics import FlatTextInput
from kivy_garden.drag_n_drop import DraggableLayoutBehavior

from ceed.utils import fix_name
from ceed.graphics import WidgetList, ShowMoreSelection, ShowMoreBehavior
from ceed.function import CeedFuncRef, FuncBase, FuncGroup, FunctionFactoryBase
from ceed.function.param_noise import NoiseBase, ParameterNoiseFactory

__all__ = ('FuncList', 'FuncWidget', 'FuncWidgetGroup', 'FuncPropTextWidget',
           'FuncNamePropTextWidget', 'FuncSettingsDropDown',
           'FuncNoiseDropDown')

_get_app = App.get_running_app

FuncWidgetType = TypeVar('FuncWidgetType', bound='FuncWidget')
FuncOrRef = Union[FuncBase, CeedFuncRef, FuncGroup]


class FuncList(DraggableLayoutBehavior, ShowMoreSelection, WidgetList,
               BoxLayout):
    """Widget that shows the list of available functions to the user and
    also allows for the creation of new functions to be added to the list.

    The functions come from :class:`ceed.function.FunctionFactoryBase`.
    """

    function_factory: FunctionFactoryBase = None
    """The :class:`ceed.function.FunctionFactoryBase` that is used for the list
    of functions available to the user and with whom new functions created in
    the GUI are registered.
    """

    is_visible = BooleanProperty(True)
    """Whether the list is currently visible.

    It is used by the selection logic and it's always True for this class.
    """

    def handle_drag_release(self, index, drag_widget):
        if drag_widget.drag_cls == 'func_spinner':
            func = self.function_factory.funcs_inst[
                drag_widget.obj_dragged.text]
        else:
            dragged = drag_widget.obj_dragged
            func = dragged.ref_func or dragged.func

        new_func = deepcopy(func)
        self.function_factory.add_func(new_func)
        self.show_func_in_listing(new_func)

    def add_func(self, name: str):
        """Adds a copy of the the function with the given ``name`` from
        :attr:`function_factory` to the available functions in
        :attr:`function_factory` (with a new name of course) or to a function
        group.
        """
        parent: Optional[FuncGroup] = None
        after: Optional[FuncBase] = None
        if self.selected_nodes:
            widget = self.selected_nodes[-1]
            if isinstance(widget, FuncWidgetGroup):
                parent = widget.func
            else:
                after = widget.func
                parent = after.parent_func

        src_func: FuncBase = self.function_factory.funcs_inst[name]

        if parent is not None:
            if parent.can_other_func_be_added(src_func):
                func = self.function_factory.get_func_ref(func=src_func)
                parent.add_func(func, after=after)
                self.show_child_func_in_func(parent, func)
        else:
            func = deepcopy(src_func)
            self.function_factory.add_func(func)
            self.show_func_in_listing(func)

    def show_child_func_in_func(
            self, parent_func: FuncGroup, child_func: FuncOrRef):
        """Displays the child function in the GUI as the child of the parent.

        :param parent_func: Function child is added to.
        :param child_func: the child function.
        """
        widget = FuncWidget.get_display_cls(child_func)()
        widget.initialize_display(child_func, self.function_factory, self)
        parent_func.display.children_container.add_widget(widget)
        if widget.expand_widget is not None:
            widget.expand_widget.state = 'down'

    def show_func_in_listing(self, func: FuncBase):
        """Shows the function, previously added to the :attr:`function_factory`
        in the GUI.

        :param func: The :class:`ceed.function.FuncBase` to show.
        """
        widget = FuncWidget.get_display_cls(func)()
        widget.initialize_display(func, self.function_factory, self)
        self.add_widget(widget)
        if widget.expand_widget is not None:
            widget.expand_widget.state = 'down'

    def get_selectable_nodes(self):
        # a ref func will never be in the root list, so get_funcs will not be
        # called on it
        return list(reversed([
            f.display for func in self.function_factory.funcs_user for
            f in func.get_funcs(step_into_ref=False) if f.display.is_visible]))

    def clear_all(self):
        """Removes all the widgets associated with the registered functions in
        :attr:`function_factory` from the GUI.
        """
        for widget in self.children[:]:
            self.remove_widget(widget)

    def show_function(self, func: FuncOrRef):
        """Displays the function (that is in the :attr:`function_factory`) in
        the GUI.

        :param func: The :class:`ceed.function.FuncBase` to display.
        """
        widget = FuncWidget.get_display_cls(func)()
        widget.initialize_display(func, self.function_factory, self)
        self.add_widget(widget)


class GroupFuncList(DraggableLayoutBehavior, BoxLayout):
    """The widget that displays the list of sub-functions contained by a
    :class:`ceed.function.FuncGroup`.
    """

    is_visible = BooleanProperty(False)
    """Whether the function list is currently expanded and visible to the user.
    """

    group_widget: 'FuncWidgetGroup' = None
    """The group function's :class:`FuncWidgetGroup` to whom this widget is
    attached to.
    """

    def handle_drag_release(self, index, drag_widget):
        group_widget = self.group_widget
        group_func = group_widget.func
        function_factory = group_func.function_factory

        # are we dragging a registered function, or some internal anonymous
        # function
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
            if widget.expand_widget is not None:
                widget.expand_widget.state = 'down'
        else:
            dragged = drag_widget.obj_dragged
            if not group_func.can_other_func_be_added(dragged.func):
                return

            func = deepcopy(dragged.func)
            group_func.add_func(func, index=len(self.children) - index)

            widget = FuncWidget.get_display_cls(func)()
            widget.initialize_display(
                func, function_factory, group_widget.selection_controller)
            self.add_widget(widget, index=index)
            if widget.expand_widget is not None:
                widget.expand_widget.state = 'down'


class FuncWidget(ShowMoreBehavior, BoxLayout):
    """The widget that represents a :class:`ceed.function.CeedFunc` instance.

    It contains all the configuration options of the function.

    The class is reused anywhere a function is shown in the GUI, including
    in :mod:`ceed.stage` so it is abstracted.
    """

    func: FuncOrRef = None
    '''The :class:`ceed.function.BaseFunc` or
    :class:`ceed.function.CeedFuncRef` instance associated with this widget.
    '''

    ref_func = None
    '''If :attr:`func` is a :class:`ceed.function.CeedFuncRef`, this is the
    actual :class:`ceed.function.BaseFunc` :attr:`func` is internally
    referencing. Otherwise, it's None.
    '''

    selected = BooleanProperty(False)
    '''Whether the function is currently selected in the GUI.
    '''

    selection_controller = None
    '''The container that gets called to select the widget when the user
    selects it with a touch. E.g. :class:`FuncList` in the function listing
    case or :class:`~ceed.stage.stage_widgets.StageFuncChildrenList` if it
    belongs to a stage.
    '''

    func_controller = ObjectProperty(None)
    '''The controller to which the function is added or removed from.
    This is e.g. :attr:`ceed.function.FunctionFactoryBase` in the function list
    case or the :class:`!ceed.stage.CeedStage` to which the function is
    attached.
    '''

    is_visible = BooleanProperty(False)
    '''Whether the function is currently visible in the GUI. I.e. when all of
    it's parents all the way to the root is visible.
    '''

    _settings = None
    """The settings widget contained in the :class:`FuncSettingsDropDown`
    representing this widget.
    """

    theme_interpolation = 0
    """The fraction by which :meth:`base_kivy_app.utils.ColorTheme.interpolate`
    interpolates the two given colors
    (:attr:`base_kivy_app.utils.ColorTheme.primary_light` and
    :attr:`base_kivy_app.utils.ColorTheme.primary`).
    """

    settings_root: 'FuncSettingsDropDown' = None
    """The :class:`FuncSettingsDropDown` used by this function to show
    settings.
    """

    expand_widget = None
    """The widget that when pressed will expand to show the :attr:`more`
    widget.
    """

    @property
    def name(self):
        """The :attr:`ceed.function.FuncBase.name` of the function.
        """
        if self.ref_func:
            return self.ref_func.name
        return self.func.name

    @staticmethod
    def get_display_cls(
            func: Union[FuncBase, FuncGroup]) -> Type[FuncWidgetType]:
        """Gets the widget class to use to display the function.

        :param func: The :class:`ceed.function.FuncBase` instance.
        :return: The widget class to use to display it in the GUI.
        """
        if isinstance(func, FuncGroup):
            return FuncWidgetGroup
        return FuncWidget

    def add_func_noise_param(self, func: FuncBase, key: str, entry_widget):
        def track_noise_param(*largs):
            entry_widget.disabled = func.noisy_parameters.get(key) is not None

        func.fbind('noisy_parameters', track_noise_param)
        track_noise_param()

        noise = Factory.NoiseSelection()
        noise.func = func
        noise.prop_name = key
        return noise

    def display_properties(self):
        """Constructs the function's configuration option widgets that is used
        by the user to customize the function in the GUI.

        It uses e.g. :meth:`ceed.function.FuncBase.get_gui_elements` and
        :meth:`ceed.function.FuncBase.get_gui_props` to get the options to
        show.
        """
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
                prop = func.property(key, True)
                if prop is None:
                    value = getattr(func, key)
                else:
                    value = prop.defaultvalue

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
                    label_w = label(
                        text=pretty_names.get(key, key), padding_x='10dp',
                        flat_color=color)
                    label_w.test_name = 'func prop label'
                    grid.add_widget(label_w)

                    value = getattr(func, key)
                    if value is True or value is False:
                        widget = FuncPropBoolWidget()
                    else:
                        widget = FuncPropTextWidget(
                            input_filter=input_filter[fmt],
                            size_hint_min_x='40dp')
                    widget.func = func
                    widget.prop_name = key
                    widget.apply_binding()
                    grid.add_widget(widget)

                    if key in noise_supported_parameters:
                        noise = self.add_func_noise_param(func, key, widget)
                    else:
                        noise = Widget(size_hint=(None, None), size=(0, 0))
                    grid.add_widget(noise)

            for key, cls in sorted(cls_widgets, key=lambda x: x[0]):
                cls, kw = cls
                if isinstance(cls, string_types):
                    cls = Factory.get(cls)

                grid.add_widget(
                    label(text=pretty_names.get(key, key),
                          padding_x='10dp', flat_color=color))
                widget = cls(func=func, prop_name=key, **kw)
                grid.add_widget(widget)

                if key in noise_supported_parameters:
                    noise = self.add_func_noise_param(func, key, widget)
                else:
                    noise = Widget(size_hint=(None, None), size=(0, 0))
                grid.add_widget(noise)

            add(grid)

        for item in items:
            if isinstance(item, string_types):
                item = Factory.get(item)()
            add(item)

    def initialize_display(
            self, func: FuncOrRef, func_controller, selection_controller):
        """Sets :attr:`selection_controller` and :attr:`func_controller`,
        and generates and applies the kv GUI rules for the widget.
        """
        if isinstance(func, CeedFuncRef):
            self.ref_func = func.func
        func.display = self
        self.func = func

        self.func_controller = func_controller  # or _get_app().function_factory
        self.selection_controller = selection_controller

        self.apply_kv()

    def remove_func(self):
        """Removes the function from its parent and removes it from the GUI.
        This is used when deleting the function in the GUI.
        """
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
        """Called from kv to hide the widget that exapnds the group view when
        the function being represented is not a group.
        """
        expand.parent.remove_widget(expand)
        self.expand_widget = None

    def replace_ref_func_with_source(self):
        """If this :attr:`func` is a :class:`ceed.function.CeedFuncRef`, this
        will replace the reference with the a copy of the original function
        being referenced and the GUI will also be updated to reflect that.
        """
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
        if widget.expand_widget is not None:
            widget.expand_widget.state = 'down'

        self.func.function_factory.return_func_ref(self.func)

    def apply_kv(self):
        """Applies the kv rules to the widget.

        The rules are manually applied to the class because we want to have
        a chance to initialize some instance variables before the kv rules is
        applied so they can be referred to from kv without having to check
        if they are None.
        """
        func = self.ref_func or self.func
        app = _get_app()
        func.fbind('on_changed', app.changed_callback)

        Builder.apply_rules(self, 'FuncWidgetStyle', dispatch_kv_post=True)

    def create_settings_dropdown(self):
        """Creates the dropdown widget that displays the function's
        configuration options.
        """
        if self.settings_root is not None:
            return

        if self.ref_func is None:
            self.settings_root = settings_root = FuncSettingsDropDown(
                func_widget=self)
            self._settings = settings_root.settings
            self.display_properties()


class FuncWidgetGroup(FuncWidget):
    """The widget used to display a :class:`ceed.function.FuncGroup` instance
    in the GUI.
    """

    children_container = None
    """The widget instance that displays all the widgets representing the
    children functions of this function.
    """

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
        if self.show_more:
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
    """The widget used to allow editing a text based configuration option of a
    :class:`ceed.function.FuncBase`.
    """

    func = None
    '''The :class:`ceed.function.FuncBase` instance it's associated with.
    '''

    prop_name = ''
    '''The name of the property of :attr:`func` that this widget edits.
    '''

    _binding = None

    def apply_binding(self):
        """Starts tracking the :attr:`prop_name` of the associated
        :attr:`func` and updates the GUI with changes.
        """
        if not self.hint_text:
            self.hint_text = self.prop_name
        uid = self.func.fbind(self.prop_name, self._update_text)
        self._binding = self.func, self.prop_name, uid
        self._update_text()

    def unbind_tracking(self):
        """Stops the tracking initialized with :meth:`apply_binding`.
        """
        if self._binding is None:
            return
        obj, prop, uid = self._binding
        obj.unbind_uid(prop, uid)
        self._binding = None

    def _update_text(self, *largs):
        """Updates the GUI whenever the :attr:`func` changes the property
        being tracked.
        """
        self.text = '{}'.format(getattr(self.func, self.prop_name))

    def _update_attr(self, text):
        """Updates the :attr:`func` property property being tracked whenever
        the GUI changes.
        """
        if not text:
            self._update_text()
            return

        if self.input_filter:
            text = {'int': int, 'float': float}[self.input_filter](text)
        setattr(self.func, self.prop_name, text)


class FuncNamePropTextWidget(FuncPropTextWidget):
    """The widget used to edit the :attr:`ceed.function.FuncBase.name` of a
    :class:`ceed.function.FuncBase`.
    """

    def _update_attr(self, text):
        if not text:
            self._update_text()
            return

        if text != self.func.name:
            self.func.name = fix_name(
                text, _get_app().function_factory.funcs_inst)


class TrackOptionsSpinner(Factory.SizedCeedFlatSpinner):
    """Similar to :class:`FuncPropTextWidget`, except this class
    allows customizing in the GUI a :attr:`func` property that can be on of a
    given list of options.
    """

    func = None
    '''The :class:`ceed.function.FuncBase` instance it's associated with.
    '''

    prop_name = ''
    '''The name of the property of :attr:`func` that this widget edits.
    '''

    allow_empty = False
    """Whether the option accepts an empty string as a valid value.
    """

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


class FuncPropBoolWidget(Factory.FlatToggleButton):
    """The widget used to control a bool configuration option of a
    :class:`ceed.function.FuncBase`.
    """

    func = None
    '''The :class:`ceed.function.FuncBase` instance it's associated with.
    '''

    prop_name = ''
    '''The name of the property of :attr:`func` that this widget edits.
    '''

    _bindings = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def update_text(*args):
            self.text = 'true' if self.state == 'down' else 'false'
        self.fbind('state', update_text)
        update_text()

    def apply_binding(self):
        """Starts tracking the :attr:`prop_name` of the associated
        :attr:`func` and updates the GUI with changes.
        """
        uid = self.func.fbind(self.prop_name, self._update_opt)
        self._bindings = [
            (self.func, self.prop_name, uid),
            (self, 'state', self.fbind('state', self._update_from_state)),
        ]
        self._update_opt()

    def unbind_tracking(self):
        """Stops the tracking initialized with :meth:`apply_binding`.
        """
        for obj, prop, uid in self._bindings:
            obj.unbind_uid(prop, uid)
        self._bindings = []

    def _update_opt(self, *largs):
        """Updates the GUI whenever the :attr:`func` changes the property
        being tracked.
        """
        self.state = 'down' if getattr(self.func, self.prop_name) else 'normal'

    def _update_from_state(self, *args):
        """Updates the :attr:`func` property property being tracked whenever
        the GUI changes.
        """
        setattr(self.func, self.prop_name, self.state == 'down')


class FuncSettingsDropDown(Factory.FlatDropDown):
    """A dropdown widget used to show settings when customizing functions.
    """

    func_widget: FuncWidget = ObjectProperty(None)
    """The :class:`FuncWidget` this dropdown is displaying settings for.
    """

    def __init__(self, func_widget, **kwargs):
        self.func_widget = func_widget
        super(FuncSettingsDropDown, self).__init__(**kwargs)


class FuncNoiseDropDown(Factory.FlatDropDown):
    """Widget for displaying options to add randomness to the functions.

    """

    noise_param: NoiseBase = ObjectProperty(None, allownone=True, rebind=True)

    noise_factory: ParameterNoiseFactory = ObjectProperty(None, rebind=True)

    func: FuncBase = None

    prop_name = ''

    param_container = None

    def __init__(self, func, prop_name, **kwargs):
        self.func = func
        self.prop_name = prop_name
        self.noise_factory = func.function_factory.param_noise_factory

        if self.prop_name in self.func.noisy_parameters:
            self.noise_param = self.func.noisy_parameters[self.prop_name]

        super(FuncNoiseDropDown, self).__init__(**kwargs)

        if self.prop_name in self.func.noisy_parameters and \
                self.noise_param is not None:
            self.show_noise_params(self.noise_param)

    def clear_noise_param(self):
        for widget in self.param_container.children[::2]:
            widget.unbind_tracking()
        self.param_container.clear_widgets()

    def set_noise_instance(self, cls_name):
        # during initial setup, the following would be the case - so return
        if self.noise_param is not None and \
                self.prop_name in self.func.noisy_parameters and \
                self.noise_param.name == cls_name and \
                self.func.noisy_parameters[self.prop_name] is self.noise_param:
            return

        self.clear_noise_param()
        self.noise_param = None
        if self.prop_name in self.func.noisy_parameters:
            del self.func.noisy_parameters[self.prop_name]

        if cls_name == 'NoNoise':
            return

        noise_param = self.noise_param = self.noise_factory.get_cls(cls_name)()
        self.func.noisy_parameters[self.prop_name] = noise_param
        self.show_noise_params(noise_param)

    def show_noise_params(self, noise_param):
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

            value = getattr(noise_param, prop)
            if value is True or value is False:
                widget = FuncPropBoolWidget()
            else:
                widget = FuncPropTextWidget(
                    input_filter='float', size_hint_min_x='40dp')
            widget.func = noise_param
            widget.prop_name = prop
            widget.apply_binding()
            grid.add_widget(widget)
