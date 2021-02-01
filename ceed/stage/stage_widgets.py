"""Stage widgets
===================

Defines the GUI components used with :mod:`ceed.stage`.
"""
from copy import deepcopy
import os
from scipy.signal import decimate
import numpy as np
from typing import Optional, Union, Type

from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty, OptionProperty
from kivy.factory import Factory
from kivy_garden.graph import MeshLinePlot
from kivy.metrics import dp
from kivy.app import App
from kivy.utils import get_color_from_hex
from kivy.lang import Builder
from kivy.graphics import Rectangle, Color, Line

from ceed.graphics import WidgetList, ShowMoreSelection, BoxSelector, \
    ShowMoreBehavior
from ceed.stage import CeedStage, StageFactoryBase, CeedStageRef, \
    last_experiment_stage_name, StageShape, CeedStageOrRefInstance
from ceed.function.func_widgets import FuncWidget, FuncWidgetGroup
from ceed.function import CeedFuncRef, FunctionFactoryBase

from kivy_garden.drag_n_drop import DraggableLayoutBehavior

__all__ = (
    'StageList', 'StageWidget', 'StageChildrenViewList', 'StageChildrenList',
    'StageFuncChildrenList', 'StageShapesChildrenList', 'StageShapeDisplay',
    'ShapePlot', 'StageGraph')


_get_app = App.get_running_app
StageOrRef = Union[CeedStage, CeedStageRef]


class StageList(DraggableLayoutBehavior, ShowMoreSelection, WidgetList,
                BoxLayout):
    """Widget that shows the list of available stages to the user and
    also allows for the creation of new stages to be added to the list.

    The functions come from :class:`ceed.stage.StageFactoryBase`.
    """

    is_visible = BooleanProperty(True)
    """Whether the list is currently visible.

    It is used by the selection logic and it's always True for this class.
    """

    stage_factory: StageFactoryBase = None
    """The :class:`ceed.stage.StageFactoryBase` that is used for the list
    of stages available to the user and with whom new stages created in
    the GUI are registered.
    """

    def __init__(self, **kwargs):
        super(StageList, self).__init__(**kwargs)
        self.nodes_order_reversed = False

    def remove_shape_from_stage(
            self, stage: CeedStage, stage_shape: StageShape):
        """This is called to remove a shape from being referenced by a stage.

        :param stage: The :class:`CeedStage` that references the shape.
        :param stage_shape: The :class:`StageShape` to be removed.
        """
        if stage_shape.display is not None:
            stage_shape.display.remove_shape()

    def add_stage(self, name: str) -> None:
        """Adds a copy of the the stage with the given ``name`` from
        :attr:`stage_factory` to the available stages in
        :attr:`stage_factory` (with a new name of course) or to a stage.
        """
        # to whom to add the stage
        parent: Optional[CeedStage] = None
        if self.selected_nodes:
            parent = self.selected_nodes[-1].stage

        src_stage: CeedStage = self.stage_factory.stage_names[name]

        if parent is not None:
            if parent.can_other_stage_be_added(src_stage):
                stage = self.stage_factory.get_stage_ref(stage=src_stage)
                parent.add_stage(stage)
                self.show_sub_stage(stage, parent)
        else:
            stage = deepcopy(src_stage)
            self.stage_factory.add_stage(stage, allow_last_experiment=False)
            self.show_stage(stage)

    def handle_drag_release(self, index, drag_widget):
        if drag_widget.drag_cls in ('stage_spinner', 'stage'):
            if drag_widget.drag_cls == 'stage_spinner':
                stage = self.stage_factory.stage_names[
                    drag_widget.obj_dragged.text]
            else:
                # we dragged a stage so we need to copy it
                stage = drag_widget.obj_dragged.stage
                if isinstance(stage, CeedStageRef):
                    stage = stage.stage

            stage = deepcopy(stage)
            self.stage_factory.add_stage(stage, allow_last_experiment=False)
            self.show_stage(stage)
            return

        # otherwise, create a new empty stage
        stage = CeedStage(
            stage_factory=self.stage_factory,
            function_factory=self.stage_factory.function_factory,
            shape_factory=self.stage_factory.shape_factory)
        self.stage_factory.add_stage(stage, allow_last_experiment=False)
        self.show_stage(stage)
        widget = stage.display

        # what did we drag, function, shape, or stage into stage?
        if drag_widget.drag_cls in ('func', 'func_spinner'):
            func_widget = StageFuncChildrenList.handle_func_drag_release(
                index, drag_widget, self, stage)
            widget.func_widget.add_widget(func_widget, index=index)
        elif drag_widget.drag_cls in ('shape', 'shape_group'):
            shape_factory = self.stage_factory.shape_factory

            if drag_widget.drag_cls == 'shape':
                item = drag_widget.obj_dragged.shape
                selection = shape_factory.selected_shapes
            elif drag_widget.drag_cls == 'shape_group':
                item = drag_widget.obj_dragged.group
                selection = shape_factory.selected_groups
            else:
                assert False

            shape = stage.add_shape(item)
            if shape is not None:
                self.show_shape_in_stage(stage, shape)

            if drag_widget.obj_dragged.selected:
                for shape in selection:
                    shape = stage.add_shape(shape)
                    if shape is not None:
                        self.show_shape_in_stage(stage, shape)
        else:
            assert False

    def get_selectable_nodes(self):
        return [d for stage in self.stage_factory.stages
                for d in stage.display.get_visible_children()]

    def clear_all(self):
        """Removes all the widgets associated with the registered stages in
        :attr:`stage_factory` from the GUI.
        """
        for widget in self.children[:]:
            self.remove_widget(widget)

    def show_stage(self, stage, expand_stage=True):
        """Displays the widget of the stage in the GUI. This is for displaying
        registered stages, not a sub-stage.

        :param stage: The :class:`CeedStage` to show.
        :param expand_stage: Whether to expand the stage options in the GUI.
            If true the user could e.g. edit the name without having to click
            first to expand the edit options.
        """
        widget = StageWidget()
        widget.initialize_display(stage, self)
        self.add_widget(widget)

        if widget.expand_widget is not None:
            widget.expand_widget.state = 'down'
            if not expand_stage:
                widget.expand_widget.state = 'normal'

    def show_sub_stage(
            self, stage: CeedStageOrRefInstance, parent_stage: CeedStage):
        """Displays the widget of the stage in the GUI. This is for displaying
        a sub-stages within the parent stage.

        :param stage: The :class:`CeedStage` to show.
        :param parent_stage: The parent :class:`CeedStage` within which to show
            the stage.
        """
        stage_widget = parent_stage.display
        widget = StageWidget()

        widget.initialize_display(stage, stage_widget.selection_controller)
        stage_widget.stage_widget.add_widget(widget)
        if widget.expand_widget is not None:
            widget.expand_widget.state = 'down'

    def show_shape_in_stage(self, stage: CeedStage, shape: StageShape):
        """Displays the widget of the shape in the GUI as belonging to the
        stage.

        :param stage: The :class:`CeedStage` to show.
        :param shape: The :class:`~ceed.stage.StageShape` to display.
        """
        shape_widget = StageShapeDisplay()
        shape_widget.initialize_display(shape, self)
        stage.display.shape_widget.add_widget(shape_widget)

    def copy_and_resample_experiment_stage(self, stage_name):
        """Makes a stage ready to be run as an experiment.

        It takes the stage, copies it and expands all sub-stages that are
        references, then samples all the parameters and finally adds this stage
        to the :attr:`stage_factory`.

        The stage is then named :attr:`ceed.stage.last_experiment_stage_name`.
        If a stage with that name already exists, that stage is first removed.

        :param stage_name: The name of the registered stage to copy.
        """
        stage = self.stage_factory.stage_names[stage_name].copy_and_resample()
        stage.name = last_experiment_stage_name

        if last_experiment_stage_name in self.stage_factory.stage_names:
            for widget in self.children:
                if widget.stage.name == last_experiment_stage_name:
                    widget.remove_stage_from_factory_no_ref()
                    break
            else:
                assert False, 'If stage is in factory it should have a widget'

        self.stage_factory.add_stage(stage)
        self.show_stage(stage, expand_stage=False)


class StageChildrenViewList(DraggableLayoutBehavior, BoxLayout):
    """The base class used by the GUI to list a stages functions, stages, or
    shapes.
    """

    is_visible = BooleanProperty(False)
    """Whether the list is currently expanded and visible to the user.
    """

    stage_widget: 'StageWidget' = None
    """The :class:`StageWidget` to whom this widget belongs.
    """


class StageChildrenList(StageChildrenViewList):
    """The container that displays the stage's sub-stage children in the GUI.
    """

    def handle_drag_release(self, index, drag_widget):
        stage_widget = self.stage_widget
        stage = stage_widget.stage
        stage_factory = stage.stage_factory

        if drag_widget.drag_cls == 'stage_spinner':
            dragged_stage = stage.stage_factory.stage_names[
                drag_widget.obj_dragged.text]
        else:
            assert drag_widget.drag_cls == 'stage'
            dragged_stage = drag_widget.obj_dragged.stage
            if isinstance(dragged_stage, CeedStageRef):
                dragged_stage = dragged_stage.stage

        assert not isinstance(stage, CeedStageRef)
        if not stage.can_other_stage_be_added(dragged_stage):
            return

        if dragged_stage.parent_stage is None:
            assert dragged_stage in stage_factory.stages

            new_stage = stage_factory.get_stage_ref(stage=dragged_stage)
        else:
            new_stage = deepcopy(dragged_stage)

        stage.add_stage(new_stage, index=len(self.children) - index)

        widget = StageWidget()
        widget.initialize_display(new_stage, stage_widget.selection_controller)
        self.add_widget(widget, index=index)
        if widget.expand_widget is not None:
            widget.expand_widget.state = 'down'


class StageFuncChildrenList(StageChildrenViewList):
    """The container that displays the stage's functions in the GUI.
    """

    @staticmethod
    def handle_func_drag_release(
            index, drag_widget, selection_controller, stage):
        """Takes a function widget being dragged into the stage and adds the
        function to the stage and creates the widget to display the function.

        :param index: The index where the widgets was dragged into.
        :param drag_widget: The widget being dragged.
        :param selection_controller: The controller that handles selection
            for the list into which the function was dragged into.
        :param stage: The associated stage.
        :return: The widget created to display the function.
        """
        stage_factory = stage.stage_factory
        function_factory = stage.function_factory

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
        else:
            func = deepcopy(drag_widget.obj_dragged.func)

        stage.add_func(func, index=len(stage.functions) - index)

        widget = FuncWidget.get_display_cls(func)()
        widget.initialize_display(func, stage, selection_controller)
        if widget.expand_widget is not None:
            widget.expand_widget.state = 'down'
        return widget

    def handle_drag_release(self, index, drag_widget):
        stage_widget = self.stage_widget
        stage = stage_widget.stage
        widget = self.handle_func_drag_release(
            index, drag_widget, stage_widget.selection_controller, stage)
        self.add_widget(widget, index=index)


class StageShapesChildrenList(StageChildrenViewList):
    """The container that displays the stage's shapes in the GUI.
    """

    def handle_drag_release(self, index, drag_widget):
        stage_widget = self.stage_widget
        stage = stage_widget.stage
        shape_factory = stage.stage_factory.shape_factory

        if drag_widget.drag_cls == 'shape':
            item = drag_widget.obj_dragged.shape
            selection = shape_factory.selected_shapes
        elif drag_widget.drag_cls == 'shape_group':
            item = drag_widget.obj_dragged.group
            selection = shape_factory.selected_groups
        else:
            assert False

        shape = stage.add_shape(item)
        if shape is not None:
            widget = StageShapeDisplay()
            widget.initialize_display(shape, stage_widget.selection_controller)
            self.add_widget(widget)

        if drag_widget.obj_dragged.selected:
            for shape in selection:
                shape = stage.add_shape(shape)
                if shape is not None:
                    widget = StageShapeDisplay()
                    widget.initialize_display(
                        shape, stage_widget.selection_controller)
                    self.add_widget(widget)


class StageWidget(ShowMoreBehavior, BoxLayout):
    """The widget displayed for an :class:`ceed.stage.CeedStage` instance.
    """

    stage: StageOrRef = None
    '''The :class:`ceed.stage.CeedStage` or :class:`ceed.stage.CeedStageRef`
    instance attached to the widget.
    '''

    ref_stage = None
    '''If :attr:`stage` is a :class:`ceed.stage.CeedStageRef`, this is the
    actual :class:`ceed.stage.CeedStage` :attr:`stage` is internally
    referencing. Otherwise, it's None.
    '''

    selected = BooleanProperty(False)
    '''Whether the stage is currently selected in the GUI.
    '''

    stage_widget: StageChildrenList = None
    '''The internal widget container to which children
    :class:`StageWidget` widget instances are added.
    '''

    func_widget: StageFuncChildrenList = None
    '''The internal widget container to which children
    :class:`ceed.func_widgets.FuncWidget` or
    :class:`ceed.func_widgets.FuncWidgetGroup` widget instances are added.
    '''

    shape_widget: StageShapesChildrenList = None
    '''The internal widget container to which children
    :class:`StageShapeDisplay` widget instances are added.
    '''

    is_visible = BooleanProperty(False)
    '''Whether the stage is currently visible in the GUI. I.e. when all of
    it's parents all the way to the root is visible.
    '''

    selection_controller = None
    '''The container that gets called to select the stage widget when the user
    selects it with a touch. E.g. :class:`StageList` in the global stage
    listing case or :class:`StageChildrenList` if it belongs to a stage.
    '''

    settings_root = None
    """The dropdown used by this function to show settings.
    """

    expand_widget = None
    """The widget that when pressed will expand to show the :attr:`more`
    widget.
    """

    @property
    def name(self):
        """The :attr:`ceed.stage.CeedStage.name` of the stage.
        """
        return self.stage.name

    def get_visible_children(self):
        """Iterates and yields all the widgets representing the sub-stages,
        functions, and shapes belonging to the stage, including the stage's
        widget itself, if they are visible. The currently collapsed/hidden
        widgets are skipped.
        """
        if not self.is_visible:
            return
        yield self

        # ref stage has no children
        if self.ref_stage is not None:
            return

        for stage in self.stage.stages:

            for child in stage.display.get_visible_children():
                if child.is_visible:
                    yield child

        for func in self.stage.functions:
            if isinstance(func, CeedFuncRef):
                if func.display.is_visible:
                    yield func.display
            else:
                for f in func.get_funcs(step_into_ref=False):
                    if f.display.is_visible:
                        yield f.display

        if self.show_more:
            for shape in self.stage.shapes:
                yield shape.display

    def initialize_display(self, stage, selection_controller):
        """Sets :attr:`selection_controller` and generates and applies the kv
        GUI rules for the widget and it's children (sub-stages, functions,
        and shapes).
        """
        if isinstance(stage, CeedStageRef):
            self.ref_stage = stage.stage
        stage.display = self
        self.stage = stage
        self.selection_controller = selection_controller

        self.apply_kv()
        if self.ref_stage:
            return

        for child_stage in stage.stages:
            display = StageWidget()
            self.stage_widget.add_widget(display)
            display.initialize_display(child_stage, selection_controller)

        for func in stage.functions:
            display = FuncWidget.get_display_cls(func)()
            self.func_widget.add_widget(display)
            display.initialize_display(func, stage, selection_controller)

        for shape in stage.shapes:
            shape_widget = StageShapeDisplay()
            shape_widget.initialize_display(shape, self.selection_controller)
            self.shape_widget.add_widget(shape_widget)

    def remove_stage_from_factory_no_ref(self):
        """Removes the stage from the :attr:`stage_factory` and GUI.

        This can only be called if the stage is not representing a
        :class:`CeedStageRef`.
        """
        self.selection_controller.clear_selection()
        assert not self.stage.parent_stage

        for item in self.get_visible_children():
            if item.selected:
                self.selection_controller.deselect_node(item)
                break

        if self.stage.stage_factory.remove_stage(self.stage):
            self.parent.remove_widget(self)

    def remove_stage(self):
        """Removes the stage from the :attr:`stage_factory` and from the GUI.
        """
        for item in self.get_visible_children():
            if item.selected:
                self.selection_controller.deselect_node(item)
                break

        self.selection_controller.clear_selection()
        if self.stage.parent_stage:
            self.stage.parent_stage.remove_stage(self.stage)
            self.parent.remove_widget(self)
        else:
            if self.stage.stage_factory.remove_stage(self.stage):
                self.parent.remove_widget(self)

        if self.ref_stage is not None:
            self.stage.stage_factory.return_stage_ref(self.stage)
        else:
            for stage in self.stage.get_stages(step_into_ref=False):
                if isinstance(stage, CeedStageRef):
                    stage.stage_factory.return_stage_ref(stage)
                else:
                    for root_func in stage.functions:
                        if isinstance(root_func, CeedFuncRef):
                            root_func.function_factory.return_func_ref(
                                root_func)
                            continue

                        for func in root_func.get_funcs(step_into_ref=False):
                            if isinstance(func, CeedFuncRef):
                                func.function_factory.return_func_ref(func)

    def replace_ref_with_source(self):
        """If this :attr:`stage` is a :class:`ceed.stage.CeedStageRef`, this
        will replace the reference with the a copy of the original stage
        being referenced and the GUI will also be updated to reflect that.
        """
        self.selection_controller.clear_selection()
        assert self.ref_stage is not None
        assert self.stage.parent_stage is not None

        parent_widget = self.parent
        parent_widget.remove_widget(self)
        stage, i = self.stage.parent_stage.replace_ref_stage_with_source(
            self.stage)

        widget = StageWidget()
        widget.initialize_display(stage, self.selection_controller)
        parent_widget.add_widget(
            widget, index=len(parent_widget.children) - i)
        if widget.expand_widget is not None:
            widget.expand_widget.state = 'down'

        self.stage.stage_factory.return_stage_ref(self.stage)

    def apply_kv(self):
        """Applies the kv rules to the widget.

        The rules are manually applied to the class because we want to have
        a chance to initialize some instance variables before the kv rules is
        applied so they can be referred to from kv without having to check
        if they are None.
        """
        app = _get_app()
        stage = self.ref_stage or self.stage
        stage.fbind('on_changed', app.changed_callback)

        Builder.apply_rules(self, 'StageWidgetStyle', dispatch_kv_post=True)

        more = self.more
        if self.ref_stage is not None:
            self.remove_widget(more)
        else:
            self.add_stage_containers(more)

    def create_settings_dropdown(self):
        """Creates the dropdown widget that displays the stage's
        configuration options.
        """
        if self.settings_root is not None or self.ref_stage is not None:
            return

        self.settings_root = Factory.FlatDropDown()
        self.settings_root.stage_widget = self
        Builder.apply_rules(
            self.settings_root, 'StageSettingsDropdownStyle',
            dispatch_kv_post=True)

    def add_stage_containers(self, more_widget):
        """Adds the widget containers for the sub-stage, functions, and shapes
        widgets of this stage.

        :param more_widget: The widget to which the containers are added. This
            widget can be made invisible in the GUI.
        """
        stage_widget = self.stage_widget = StageChildrenList()
        stage_widget.stage_widget = self
        stage_widget.drag_classes = ['stage', 'stage_spinner']
        stage_widget.drag_append_end = False
        stage_widget.back_color = (.482, .114, 0, 1)
        more_widget.add_widget(stage_widget)
        Builder.apply_rules(
            stage_widget, 'StageContainerListStyle', dispatch_kv_post=True)

        func_widget = self.func_widget = StageFuncChildrenList()
        func_widget.stage_widget = self
        func_widget.drag_classes = ['func', 'func_spinner']
        func_widget.drag_append_end = False
        func_widget.back_color = (.196, .122, .063, 1)
        more_widget.add_widget(func_widget)
        Builder.apply_rules(
            func_widget, 'StageContainerListStyle', dispatch_kv_post=True)

        shape_widget = self.shape_widget = StageShapesChildrenList()
        shape_widget.stage_widget = self
        shape_widget.drag_classes = ['shape', 'shape_group']
        shape_widget.drag_append_end = True
        shape_widget.back_color = (.835, .278, 0, 1)
        more_widget.add_widget(shape_widget)
        Builder.apply_rules(
            shape_widget, 'StageContainerListStyle', dispatch_kv_post=True)

        stage_widget.test_name = 'stage child list'
        func_widget.test_name = 'stage func list'
        shape_widget.test_name = 'stage shape list'


class StageShapeDisplay(BoxSelector):
    """The widget used to display a :class:`ceed.stage.StageShape`
    representation of the shape in the stage.
    """

    stage_shape: StageShape = None
    '''The :class:`ceed.stage.StageShape` instance that this widget displays.
    '''

    selected = BooleanProperty(False)
    """Whether the widget is currently selected.
    Read only.
    """

    selection_controller = None
    '''The container that gets called to select the shape widget when the user
    selects it with a touch.
    '''

    is_visible = BooleanProperty(True)
    """Whether the shape is currently visible in the stage shape's list.
    """

    settings_root = None
    """The dropdown configuring the shape.
    """

    @property
    def name(self):
        """The :attr:`ceed.stage.StageShape.name` of the shape or shape group.
        """
        return self.stage_shape.name

    def initialize_display(self, stage_shape, selection_controller):
        """Sets :attr:`selection_controller` and generates and applies the kv
        GUI rules for the widget (``StageShapeDisplayStyle``).
        """
        stage_shape.display = self
        self.stage_shape = stage_shape
        self.controller = self.selection_controller = selection_controller

        Builder.apply_rules(
            self, 'StageShapeDisplayStyle', dispatch_kv_post=True)

    def remove_shape(self):
        """Removes the shape from being referenced by the stage and also
        removes it from the GUI.
        """
        self.stage_shape.stage.remove_shape(self.stage_shape)
        if self.selected:
            self.selection_controller.deselect_node(self)
        self.parent.remove_widget(self)

    def create_settings_dropdown(self):
        """Creates the dropdown widget that displays the shape's
        configuration options.
        """
        if self.settings_root is not None:
            return

        self.settings_root = Factory.FlatDropDown()
        self.settings_root.stage_shape = self.stage_shape
        Builder.apply_rules(
            self.settings_root, 'StageShapeDropdownStyle',
            dispatch_kv_post=True)


class ShapePlot(object):
    '''A plot that displays the time-intensity series of a shape in r, g, b
    seperately.
    '''

    r_plot = None
    '''The :class:`MeshLinePlot` used by the red color.
    '''

    g_plot = None
    '''The :class:`MeshLinePlot` used by the green color.
    '''

    b_plot = None
    '''The :class:`MeshLinePlot` used by the blue color.
    '''

    r_btn = None
    '''The button in the GUI that controls whether the red color is displayed.
    '''

    g_btn = None
    '''The button in the GUI that controls whether the green color is
    displayed.
    '''

    b_btn = None
    '''The button in the GUI that controls whether the blue color is displayed.
    '''

    selection_label = None
    '''The label that shows the plot name next to the color selection.
    '''

    plot_label = None
    '''The label used on the plot to label the plot.
    '''

    name = ''
    '''The name of the shape displayed by this plot.
    '''

    color_values = None
    '''A Tx4 numpy array with the values r, g, b, a for all time points.
    '''

    graph_canvas = None
    '''The canvas to which the plot graphics instructions are added.
    '''

    frame_rate = 30.
    '''The sampling rate used to sample the plot across time.
    '''

    graph = None
    '''The :class:`StageGraph` that displays this p;ot.
    '''

    background = []
    '''A list that contains the color graphics instructions used to color
    the back of each plot.
    '''

    def __init__(self, name='', graph_canvas=None, graph=None, **kwargs):
        super(ShapePlot, self).__init__(**kwargs)
        self.name = name
        self.graph = graph
        self.graph_canvas = graph_canvas
        self.background = []
        r = self.r_btn = Factory.ShapeGraphSelector(text='R')
        r.controller = graph
        g = self.g_btn = Factory.ShapeGraphSelector(text='G')
        g.controller = graph
        b = self.b_btn = Factory.ShapeGraphSelector(text='B')
        b.controller = graph
        app = App.get_running_app()
        self.selection_label = Factory.FlatMinXYSizedLabel(
            text=name, padding=(dp(5), dp(5)),
            flat_color=app.theme.text_primary)
        self.plot_label = Factory.FlatXMinYSizedLabel(
            text=name, padding=(dp(5), dp(5)),
            flat_color=app.theme.text_primary)

    def update_plot_instructions(self):
        '''Updates the graphics instructions when the plot is shown/hidden.
        Returns True when the plot was shown/hidden and False otherwise.
        '''
        changed = False
        vals = self.color_values
        rate = self.frame_rate
        for i, chan in enumerate('rgb'):
            active = getattr(self, '{}_btn'.format(chan)).state == 'down'
            plot_attr = '{}_plot'.format(chan)

            if active and not getattr(self, plot_attr):
                if not self.r_plot and not self.g_plot and not self.b_plot:
                    with self.graph_canvas:
                        c = Color(*get_color_from_hex('ebebeb'))
                        r = Line()
                        self.background = [r, c]

                color = [0, 0, 0, 1]
                color[i] = 1

                plot = MeshLinePlot(color=color)
                plot.params['ymin'] = 0
                plot.params['ymax'] = 1
                add = self.graph_canvas.add
                for instr in plot.get_drawings():
                    add(instr)
                plot.ask_draw()

                setattr(self, plot_attr, plot)
                changed = True
            elif not active and getattr(self, plot_attr):
                changed = True
                plot = getattr(self, plot_attr)
                remove = self.graph_canvas.remove

                for instr in plot.get_drawings():
                    remove(instr)

                setattr(self, plot_attr, None)

                if not self.r_plot and not self.g_plot and not self.b_plot:
                    for instr in self.background:
                        remove(instr)
                    self.background = []

        return changed

    def remove_plot(self):
        '''Removes the r, g, b plots from the graph and hides them.
        '''
        for chan in 'rgb':
            plot_attr = '{}_plot'.format(chan)
            plot = getattr(self, plot_attr)

            if plot:
                remove = self.graph_canvas.remove
                for instr in plot.get_drawings():
                    remove(instr)
                setattr(self, plot_attr, None)

    def update_plot_params(self, rect, xmin, xmax, start, end, time, factors):
        '''Updates the paraeters of the plot, e.g. the time range displayed
        etc.

        :Params:

            `rect`: 4-tuple
                ``(x1, y1, x2, y2)``, where ``x1``, ``y1``, is the position
                of the lower-left corner of the screen area that displays the
                plot and ``x2``, ``y2`` is the upper-right position.
            `xmin`: float
                Similar to :attr:`StageGraph.xmin`.
            `xmax`: float
                Similar to :attr:`StageGraph.xmax`.
            `start`: int
                The index in time in :attr:`color_values` from where to display
                the data.
            `end`: int
                The index in time in :attr:`color_values` until where to
                display the data.
            `time`: numpy array
                A numpy array with the time values of the for the data to be
                displayed.
            `factors`: list of ints
                A list of successive factors by which :attr:`color_values`
                needs to be decimated so that it'll match ``time``. We
                down-sample when the data is too large.
        '''
        plots = []
        active = []
        for plot in (self.r_plot, self.g_plot, self.b_plot):
            active.append(bool(plot))
            if plot:
                plots.append(plot)

        vals = self.color_values[start:end, :3][:, active]
        for factor in factors:
            vals = decimate(vals, factor, axis=0, zero_phase=True)

        for i, plot in enumerate(plots):
            params = plot.params
            if params['xmin'] != xmin:
                params['xmin'] = xmin
            if params['xmax'] != xmax:
                params['xmax'] = xmax

            plot.points = [(time[j], vals[j, i]) for j in range(len(time))]

        self.update_plot_sizing(rect)

    def update_plot_sizing(self, rect):
        '''Updates only the physical position of the plot on screen.

        :Params:

            `rect`: 4-tuple
                ``(x1, y1, x2, y2)``, where ``x1``, ``y1``, is the position
                of the lower-left corner of the screen area that displays the
                plot and ``x2``, ``y2`` is the upper-right position.
        '''
        for plot in (self.r_plot, self.g_plot, self.b_plot):
            if not plot:
                continue

            params = plot.params
            if params['size'] != rect:
                params['size'] = rect
                line = self.background[0]
                x1, y1, x2, y2 = rect
                line.points = [x1, y2, x1, y1, x2, y1]

    def force_update(self):
        '''Simply forces a re-draw of the plot using the last settings.
        '''
        for plot in (self.r_plot, self.g_plot, self.b_plot):
            if not plot:
                continue
            plot.ask_draw()


class StageGraph(Factory.FlatSplitter):
    '''Displays a time-intensity plot for all the shapes of a
    :class:`ceed.stage.CeedStage`.
    '''

    plot_values = ObjectProperty(None)
    '''The computed intensity values for the shapes for all times as returned
    by :meth:`~ceed.stage.StageFactoryBase.get_all_shape_values`.
    '''

    plots = DictProperty({})
    '''A dict whose keys are the name of shapes and whose values are the
    :class:`ShapePlot` instances visualizing the shape.
    '''

    n_plots_displayed = NumericProperty(0)
    '''The number of :class:`ShapePlot` currently displayed. Read-only.
    '''

    shape_height = NumericProperty(dp(40))
    '''The height of each plot.
    '''

    shape_spacing = NumericProperty(dp(5))
    '''The spacing between plots.
    '''

    xmin = NumericProperty(0)
    '''The min time of the whole data set. This is the start time of the data.
    '''

    xmax = NumericProperty(1)
    '''The max time of the whole data set. This is the end time of the data.
    '''

    view_xmin = NumericProperty(0)
    '''The visible start time of the graph being displayed.
    '''

    view_xmax = NumericProperty(1)
    '''The visible end time of the graph being displayed.
    '''

    r_selected = OptionProperty('none', options=['all', 'none', 'some'])
    '''Which of the shape's plots is currently shown for the red color.

    Can be one of ``'all'``, ``'none'`` ``'some'``.
    '''

    g_selected = OptionProperty('none', options=['all', 'none', 'some'])
    '''Which of the shape's plots is currently shown for the green color.

    Can be one of ``'all'``, ``'none'`` ``'some'``.
    '''

    b_selected = OptionProperty('none', options=['all', 'none', 'some'])
    '''Which of the shape's plots is currently shown for the blue color.

    Can be one of ``'all'``, ``'none'`` ``'some'``.
    '''

    time_points = []
    '''The list of time points for which the shapes have an intensity value.
    It's in the range of (:attr:`xmin`, :attr:`xmax`). It is automatically set
    by :meth:`refresh_graph`.
    '''

    frame_rate = 1.
    '''The sampling rate used to sample the shape intensity values from the
    functions in time. It is automatically set by :meth:`refresh_graph`.
    '''

    def __init__(self, **kwargs):
        super(StageGraph, self).__init__(**kwargs)
        self._shapes_displayed_update_trigger = Clock.create_trigger(
            self.sync_plots_shown)

        self._plot_pos_update_trigger = Clock.create_trigger(
            self.refresh_plot_pos)
        self.fbind('xmin', self._plot_pos_update_trigger)
        self.fbind('xmax', self._plot_pos_update_trigger)
        self.fbind('view_xmin', self._plot_pos_update_trigger)
        self.fbind('view_xmax', self._plot_pos_update_trigger)
        self.fbind('shape_height', self._plot_pos_update_trigger)
        self.fbind('shape_spacing', self._plot_pos_update_trigger)

        self._plot_sizing_update_trigger = Clock.create_trigger(
            self.refresh_plot_sizing)

    @property
    def sorted_plots(self):
        '''A list of tuples of the key, value pairs of :attr:`plots` sorted
        by shape (plot) name.
        '''
        return sorted(self.plots.items(), key=lambda x: x[0])

    def refresh_graph(self, stage_name: str, frame_rate):
        '''Re-samples the intensity values for the shapes from the stage.

        :Params:

            `stage_name`: str
                The stage from which to sample the shape intensity values.
            `frame_rate`: float
                The sampling rate used to sample the shape intensity values
                from the functions in time.

        '''
        factory = App.get_running_app().stage_factory
        stage: CeedStage = factory.stage_names[stage_name]
        stage = stage.copy_and_resample()

        frame_rate = self.frame_rate = float(frame_rate)
        vals = self.plot_values = factory.get_all_shape_values(
            frame_rate, stage=stage,
            pre_compute=App.get_running_app().view_controller.
            pre_compute_stages)
        N = len(list(vals.values())[0]) if vals else 0

        plots = self.plots
        names = set(self.plot_values.keys())
        changed = False

        for name in list(plots.keys()):
            plots[name].remove_plot()
            if name not in names:
                del plots[name]
                changed = True

        for name in names:
            if name not in plots:
                plots[name] = ShapePlot(
                    name=name, graph_canvas=self.graph.canvas, graph=self)
                changed = True
            plots[name].color_values = np.array(vals[name])
            plots[name].frame_rate = frame_rate

        self.time_points = np.arange(N, dtype=np.float64) / float(frame_rate)
        self.view_xmin = self.xmin = 0
        self.view_xmax = self.xmax = max(N, 1) / frame_rate

        self._shapes_displayed_update_trigger()

        if changed:
            self.shape_selection_widget.clear_widgets()
            add = self.shape_selection_widget.add_widget
            for _, plot in self.sorted_plots:
                add(plot.selection_label)
                add(plot.r_btn)
                add(plot.g_btn)
                add(plot.b_btn)

    def sync_plots_shown(self, *largs):
        '''Checks which plots were selected/deselected in the GUI and
        shows/hides the corresponding plots.
        '''
        pos_changed = False
        n = len(self.plots)
        r = g = b = 0
        i = 0
        for _, plot in self.sorted_plots:
            # checks the button for whether the plots are shown
            pos_changed = plot.update_plot_instructions() or pos_changed

            if plot.r_plot:
                r += 1
            if plot.g_plot:
                g += 1
            if plot.b_plot:
                b += 1

            if plot.r_plot or plot.g_plot or plot.b_plot:
                i += 1

        self.n_plots_displayed = i
        self.r_selected = 'none' if not r else ('all' if n == r else 'some')
        self.g_selected = 'none' if not g else ('all' if n == g else 'some')
        self.b_selected = 'none' if not b else ('all' if n == b else 'some')

        if pos_changed:
            self._plot_pos_update_trigger()
            self.graph_labels.clear_widgets()
            add = self.graph_labels.add_widget
            for _, plot in self.sorted_plots:
                if plot.r_plot or plot.g_plot or plot.b_plot:
                    add(plot.plot_label)

    def apply_selection_all(self, channel):
        '''Hides or shows all the plots for a color ``channel``, e.g. red,
        depending on the state of the corresponding :attr:`` r_selected.

        It cycles between selecting all and derselecting all of that channel.

        :params:

            `channel`: str
                One of ``'r'``,  ``'g'``, or  ``'b'``.
        '''
        btn_attr = '{}_btn'.format(channel)
        state = getattr(self, '{}_selected'.format(channel))
        down = state == 'none' or state == 'some'
        for plot in self.plots.values():
            getattr(plot, btn_attr).state = 'down' if down else 'normal'
        self._shapes_displayed_update_trigger()

    def refresh_plot_pos(self, *largs):
        '''Called when any of the plot parameters, e.g. :attr:`xmax` or
        :attr:`view_xmax`, changes and the plots needs to be updated.
        '''
        spacing = self.shape_spacing
        plot_h = self.shape_height
        xmin, xmax = self.view_xmin, self.view_xmax
        x, y = self.graph.pos
        w = self.graph.width
        factors = []

        s = max(0, int(np.ceil((xmin - self.xmin) * self.frame_rate)))
        e = min(int(np.floor((xmax - self.xmin) * self.frame_rate)) + 1,
                len(self.time_points))

        time = self.time_points[s:e]
        N = e - s

        while N > 2048:
            factor = np.floor(N / 2048.)
            if factor <= 1.5:
                break

            factor = min(10, int(np.ceil(factor)))
            factors.append(factor)
            time = decimate(time, factor, zero_phase=True)
            N = len(time)

        i = 0
        for _, plot in reversed(self.sorted_plots):
            if not plot.r_plot and not plot.g_plot and not plot.b_plot:
                continue

            rect = (x, y + i * (plot_h + spacing),
                    x + w, y + (i + 1) * plot_h + i * spacing)
            plot.update_plot_params(rect, xmin, xmax, s, e, time, factors)
            i += 1

    def refresh_plot_sizing(self, *largs):
        '''Called when the physical position of the graph needs to be updated
        but none of the parameters of the graph has changed.
        '''
        spacing = self.shape_spacing
        plot_h = self.shape_height
        x, y = self.graph.pos
        w = self.graph.width

        i = 0
        for _, plot in reversed(self.sorted_plots):
            if not plot.r_plot and not plot.g_plot and not plot.b_plot:
                continue

            rect = (x, y + i * (plot_h + spacing),
                    x + w, y + (i + 1) * plot_h + i * spacing)
            plot.update_plot_sizing(rect)
            i += 1

    def set_pin(self, state):
        '''Switches between the graph being displayed as a pop-up and as
        inlined in the app.

        :Params:

            `state`: bool
                When True, the graph will be displayed inlined in the app.
                When False it is displayed as pop-up.
        '''
        self.parent.remove_widget(self)
        if state:
            App.get_running_app().pinned_graph.add_widget(self)
            self.unpinned_root.dismiss()
        else:
            self.unpinned_parent.add_widget(self)
            self.unpinned_root.open()
