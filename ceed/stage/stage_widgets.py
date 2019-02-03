'''Stage widgets
===================

Defines the GUI components used with :mod:`ceed.stage`.
'''
from copy import deepcopy
import os
if not os.environ.get('KIVY_DOC_INCLUDE', None):
    from scipy.signal import decimate
import numpy as np

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty, OptionProperty
from kivy.factory import Factory
from kivy.garden.graph import MeshLinePlot
from kivy.metrics import dp
from kivy.app import App
from kivy.utils import get_color_from_hex
from kivy.lang.compiler import kv, KvContext, KvRule
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Rectangle, Color, Line

from ceed.graphics import WidgetList, ShowMoreSelection, BoxSelector, \
    ShowMoreBehavior
from ceed.stage import CeedStage, CeedStageRef
from ceed.function.func_widgets import FuncWidget, FuncWidgetGroup, CeedFuncRef

from cplcom.drag_n_drop import DraggableLayoutBehavior

__all__ = ('StageList', 'StageWidget', 'StageShapeDisplay', 'ShapePlot',
           'StageGraph')


_get_app = App.get_running_app


class StageList(DraggableLayoutBehavior, ShowMoreSelection, WidgetList,
                BoxLayout):
    '''Widget that shows the list of all the stages.
    '''

    is_visible = BooleanProperty(True)

    paint_controller = None

    stage_factory = None

    def __init__(self, **kwargs):
        super(StageList, self).__init__(**kwargs)
        self.nodes_order_reversed = False

    def remove_shape_from_stage(self, stage, stage_shape):
        self.clear_selection()
        stage.remove_shape(stage_shape)
        display = stage_shape.display
        if display is not None:
            display.parent.remove_widget(display)

    def add_empty_stage(self):
        stage = self.stage_factory.make_stage({'cls': 'CeedStage'})
        widget = StageWidget()

        if self.selected_nodes:
            stage_widget = self.selected_nodes[-1]
            stage_widget.stage.add_stage(stage)

            widget.initialize_display(stage, stage_widget.selection_controller)
            stage_widget.stage_widget.add_widget(widget)
        else:
            self.stage_factory.add_stage(stage)
            widget.initialize_display(stage, self)
            self.add_widget(widget)
        return stage

    def handle_drag_release(self, index, drag_widget):
        if drag_widget.drag_cls == 'stage':
            stage = drag_widget.obj_dragged.stage
            if isinstance(drag_widget.obj_dragged.stage, CeedStageRef):
                stage = stage.stage
            stage = deepcopy(stage)

            self.stage_factory.add_stage(stage)

            widget = StageWidget()
            widget.initialize_display(stage, self)
            self.add_widget(widget)
            return

        stage = self.stage_factory.make_stage({'cls': 'CeedStage'})
        self.stage_factory.add_stage(stage)

        widget = StageWidget()
        widget.initialize_display(stage, self)
        self.add_widget(widget)

        if drag_widget.drag_cls in ('func', 'func_spinner'):
            func_widget = StageFuncChildrenList._handle_drag_release(
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
                shape_widget = StageShapeDisplay()
                shape_widget.initialize_display(shape, self)
                widget.shape_widget.add_widget(shape_widget)

            if drag_widget.obj_dragged.selected:
                for shape in selection:
                    shape = stage.add_shape(shape)
                    if shape is not None:
                        shape_widget = StageShapeDisplay()
                        shape_widget.initialize_display(shape, self)
                        widget.shape_widget.add_widget(shape_widget)
        else:
            assert False

    def get_selectable_nodes(self):
        return [d for stage in self.stage_factory.stages
                for d in stage.display.get_visible_children()]

    def clear_all(self):
        for widget in self.children[:]:
            self.remove_widget(widget)

    def show_stage(self, stage):
        widget = StageWidget()
        widget.initialize_display(stage, self)
        self.add_widget(widget)


class StageChildrenViewList(DraggableLayoutBehavior, BoxLayout):

    is_visible = BooleanProperty(False)

    stage_widget = None


class StageChildrenList(StageChildrenViewList):

    def handle_drag_release(self, index, drag_widget):
        stage_widget = self.stage_widget
        stage = stage_widget.stage
        stage_factory = stage.stage_factory
        dragged_stage = drag_widget.obj_dragged.stage

        assert not isinstance(stage, CeedStageRef)
        if not stage.can_other_stage_be_added(dragged_stage):
            return

        assert drag_widget.drag_cls == 'stage'
        if stage.parent_stage is None:
            assert stage in stage_factory.stages

            new_stage = stage_factory.get_stage_ref(stage=dragged_stage)
        else:
            if not stage.can_other_stage_be_added(dragged_stage):
                return

            new_stage = deepcopy(dragged_stage)

        stage.add_stage(new_stage, index=len(self.children) - index)

        widget = StageWidget()
        widget.initialize_display(new_stage, stage_widget.selection_controller)
        self.add_widget(widget, index=index)


class StageFuncChildrenList(StageChildrenViewList):

    @staticmethod
    def _handle_drag_release(index, drag_widget, selection_controller, stage):
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
        return widget

    def handle_drag_release(self, index, drag_widget):
        stage_widget = self.stage_widget
        stage = stage_widget.stage
        widget = self._handle_drag_release(
            index, drag_widget, stage_widget.selection_controller, stage)
        self.add_widget(widget, index=index)


class StageShapesChildrenList(StageChildrenViewList):

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
    '''The widget displayed for an :class:`ceed.stage.CeedStage` instance.
    '''

    stage = None
    '''The :class:`ceed.stage.CeedStage` instance attached to the widget.
    '''

    ref_stage = None

    selected = BooleanProperty(False)

    stage_widget = None
    '''The internal widget container to which children
    :class:`StageWidget` widget instances are added.
    '''

    func_widget = None
    '''The internal widget container to which children
    :class:`ceed.func_widgets.FuncWidget` or
    :class:`ceed.func_widgets.FuncWidgetGroup` widget instances are added.
    '''

    shape_widget = None
    '''The internal widget container to which children
    :class:`StageShapeDisplay` widget instances are added.
    '''

    is_visible = BooleanProperty(False)

    children_container = None

    selection_controller = None

    @property
    def name(self):
        '''The :attr:`ceed.stage.CeedStage.name` of the stage.
        '''
        return self.stage.name

    def get_visible_children(self):
        if not self.is_visible:
            return
        yield self

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

    def remove_stage(self):
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

        for item in self.get_visible_children():
            if item.selected:
                self.selection_controller.deselect_node(item)
                break

    def replace_ref_with_source(self):
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

        self.stage.stage_factory.return_stage_ref(self.stage)

    @kv(proxy='app*')
    def apply_kv(self):
        app = _get_app()
        stage = self.ref_stage or self.stage
        stage.fbind('on_changed', app.changed_callback)

        with KvContext():
            self.size_hint_y = None
            self.orientation = 'vertical'
            self.spacing = '3dp'

            self.height @= self.minimum_height
            self.size_hint_min_x @= self.minimum_width
            self.is_visible @= self.parent is not None and self.parent.is_visible

            with BoxSelector(
                    parent=self, size_hint_y=None, height='34dp',
                    orientation='horizontal', spacing='5dp', padding='5dp'
                    ) as selector:
                selector.size_hint_min_x @= selector.minimum_width
                selector.controller = self.selection_controller

                with selector.canvas:
                    color = Color()
                    color.rgba ^= app.theme.interpolate(
                        app.theme.primary_light, app.theme.primary, .4) if \
                        not self.selected else app.theme.primary
                    rect = Rectangle()
                    rect.size ^= selector.size
                    rect.pos ^= selector.pos

                with Factory.DraggingWidget(
                        parent=selector, drag_widget=selector,
                        obj_dragged=self, drag_cls='stage') as dragger:
                    dragger.drag_copy = True  # root.func.parent_func is None
                    dragger.flat_color = .482, .114, 0, 1
                    # if not self.drag_copy: root.remove_func()
                    pass

                with Factory.ExpandWidget(parent=selector) as expand:
                    expand.state = 'down'
                    self.show_more @= expand.is_open
                if self.ref_stage is not None:
                    selector.remove_widget(expand)

                with Factory.FlatLabel(
                        parent=selector, center_texture=False,
                        padding=('5dp', '5dp')) as stage_label:
                    stage_label.flat_color = app.theme.text_primary

                    stage_label.text @= stage.name
                    stage_label.size_hint_min_x @= stage_label.texture_size[0]

                with Factory.FlatImageButton(
                        parent=selector, scale_down_color=True,
                        source='flat_delete.png') as del_btn:
                    del_btn.flat_color @= app.theme.accent
                    del_btn.disabled @= stage.has_ref if self.ref_stage is None else False
                    with KvRule(del_btn.on_release, triggered_only=True):
                        self.remove_stage()

                if self.ref_stage:
                    with KvContext():
                        with Factory.FlatImageButton(
                                parent=selector, scale_down_color=True,
                                source='call-split.png') as split_btn:
                            split_btn.flat_color @= app.theme.accent

                            with KvRule(split_btn.on_release, triggered_only=True):
                                self.replace_ref_with_source()
                else:
                    with KvContext():
                        with Factory.FlatImageButton(
                                parent=selector, scale_down_color=True,
                                source='flat_dots_vertical.png') as more_btn:
                            more_btn.flat_color @= app.theme.accent

                            settings_root, splitter = self.get_settings_dropdown()
                            with KvRule(more_btn.on_release, triggered_only=True):
                                assert self.ref_stage is None
                                settings_root.open(selector)
                                splitter.width = max(selector.width, splitter.width)

            with BoxLayout(
                    parent=self, spacing='3dp', size_hint_y=None,
                    orientation='vertical', padding=('15dp', 0, 0, 0)) as more:
                self.more = more
                if self.ref_stage is not None:
                    self.remove_widget(more)

                more.height @= more.minimum_height
                more.size_hint_min_x @= more.minimum_width
                self.stage_widget = self.add_children_container(
                    more, ['stage'], StageChildrenList,
                    (.482, .114, 0, 1))
                self.func_widget = self.add_children_container(
                    more, ['func', 'func_spinner'], StageFuncChildrenList,
                    (.196, .122, .063, 1))
                self.shape_widget = self.add_children_container(
                    more, ['shape', 'shape_group'], StageShapesChildrenList,
                    (.835, .278, 0, 1), True)

    @kv(proxy='app*')
    def get_settings_dropdown(self):
        """Creates the dropdown with the settings.
        """
        assert self.ref_stage is None
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
                    settings.height @= settings.minimum_height
                    settings.size_hint_min_x @= settings.minimum_width

                    with Factory.FlatSizedTextInput(parent=settings) as name_input:
                        name_input.background_color @= app.theme.primary_text
                        name_input.text @= self.stage.name
                        with KvRule(name_input.focus):
                            if not name_input.focus:
                                self.stage.name = name_input.text
                    if self.stage.parent_stage is not None:
                        settings.remove_widget(name_input)

                    with BoxLayout(
                            parent=settings, size_hint_y=None, height='34dp',
                            spacing='5dp') as channel_box:

                        with Factory.LightThemedToggleButton(
                                parent=channel_box, text='R') as channel_r:
                            channel_r.state @= 'down' if self.stage.color_r else 'normal'
                            self.stage.color_r @= channel_r.state == 'down'
                        with Factory.LightThemedToggleButton(
                                parent=channel_box, text='G') as channel_g:
                            channel_g.state @= 'down' if self.stage.color_g else 'normal'
                            self.stage.color_g @= channel_g.state == 'down'
                        with Factory.LightThemedToggleButton(
                                parent=channel_box, text='B') as channel_b:
                            channel_b.state @= 'down' if self.stage.color_b else 'normal'
                            self.stage.color_b @= channel_b.state == 'down'

                    with GridLayout(
                            parent=settings, size_hint_y=None, padding='5dp',
                            spacing='5dp', cols=2) as grid:
                        grid.height @= grid.minimum_height
                        grid.size_hint_min_x @= grid.minimum_width
                        with Factory.FlatLabel(
                            parent=grid, size_hint=(None, None),
                                text='Stage order') as stage_label:
                            stage_label.size @= stage_label.texture_size
                            stage_label.flat_color @= app.theme.text_primary

                        with BoxLayout(
                                parent=grid, spacing='5dp', size_hint_y=None
                                    ) as order_box:
                            order_box.height @= order_box.minimum_height
                            order_box.size_hint_min_x @= order_box.minimum_width

                            with Factory.LightThemedToggleButton(
                                    parent=order_box, size_hint_y=None,
                                    padding=('5dp', '5dp'), text='Serial'
                                    ) as serial:
                                serial.height @= serial.texture_size[1]
                                serial.size_hint_min_x @= serial.texture_size[0]

                                serial.state @= 'down' if self.stage.order == 'serial' else 'normal'
                                with KvRule(serial.state):
                                    self.stage.order = 'serial' if serial.state == 'down' else 'parallel'

                            with Factory.LightThemedToggleButton(
                                    parent=order_box, size_hint_y=None,
                                    padding=('5dp', '5dp'), text='Parallel'
                                    ) as parallel:
                                parallel.height @= parallel.texture_size[1]
                                parallel.size_hint_min_x @= parallel.texture_size[0]

                                parallel.state @= 'down' if self.stage.order == 'parallel' else 'normal'
                                with KvRule(parallel.state):
                                    self.stage.order = 'parallel' if parallel.state == 'down' else 'serial'

                        with Factory.FlatLabel(
                                parent=grid, size_hint=(None, None),
                                text='End on') as end_label:
                            end_label.size @= end_label.texture_size
                            end_label.flat_color @= app.theme.text_primary
                        with BoxLayout(
                                parent=grid, spacing='5dp', size_hint_y=None
                                ) as end_box:
                            end_box.height @= end_box.minimum_height
                            end_box.size_hint_min_x @= end_box.minimum_width

                            with Factory.LightThemedToggleButton(
                                    parent=end_box, size_hint_y=None,
                                    padding=('5dp', '5dp'), text='All') as end_all:
                                end_all.height @= end_all.texture_size[1]
                                end_all.size_hint_min_x @= end_all.texture_size[0]
                                end_all.state @= 'down' if self.stage.complete_on == 'all' else 'normal'
                                with KvRule(end_all.state):
                                    self.stage.complete_on = 'all' if end_all.state == 'down' else 'any'

                            with Factory.LightThemedToggleButton(
                                    parent=end_box, size_hint_y=None,
                                    padding=('5dp', '5dp'), text='Any') as end_any:
                                end_any.height @= end_any.texture_size[1]
                                end_any.size_hint_min_x @= end_any.texture_size[0]
                                end_any.state @= 'down' if self.stage.complete_on == 'any' else 'normal'
                                with KvRule(end_any.state):
                                    self.stage.complete_on = 'any' if end_any.state == 'down' else 'all'

        return settings_root, splitter

    @kv(proxy='app*')
    def add_children_container(
            self, container, drag_classes, cls, color, drag_append_end=False):
        """Gets a StageChildrenList that is added to containter.
        """
        app = _get_app()
        with KvContext():
            with cls(parent=container) as widget:
                widget.spacing = '5dp'
                widget.size_hint_y = None
                widget.orientation = 'vertical'
                widget.spacer_props = {
                    'size_hint_y': None, 'height': '40dp',
                    'size_hint_min_x': '40dp'}
                widget.drag_classes = drag_classes
                widget.drag_target_stage = self.stage
                widget.drag_append_end = drag_append_end

                widget.height @= widget.minimum_height
                widget.size_hint_min_x @= widget.minimum_width
                widget.padding @= '5dp', 0, 0, (
                    0 if widget.children and not (
                        app.drag_controller.dragging and
                        app.drag_controller.widget_dragged and
                        app.drag_controller.widget_dragged.drag_cls in
                        drag_classes)
                    else '12dp')
                widget.is_visible @= self.show_more and self.is_visible
                widget.stage_widget = self

                with widget.canvas:
                    color_back = Color()
                    color_back.rgba = 152 / 255., 153 / 255., 155 / 255., 1.

                    rect_back = Rectangle()
                    rect_back.pos ^= widget.x + dp(5), widget.y
                    rect_back.size ^= widget.width - dp(5), dp(10) if (
                        app.drag_controller.dragging and
                        app.drag_controller.widget_dragged and
                        app.drag_controller.widget_dragged.drag_cls in
                        drag_classes) else 0

                    color_inst = Color()
                    color_inst.rgba = color
                    rect = Rectangle()
                    rect.pos ^= widget.x + dp(1), widget.y
                    rect.size ^= dp(2), widget.height
        return widget


class StageShapeDisplay(BoxSelector):
    '''The widget used for the :class:`ceed.stage.StageShape`.
    '''

    stage_shape = None
    '''The :class:`ceed.stage.StageShape` instance that this widget displays.
    '''

    selected = BooleanProperty(False)

    selection_controller = None

    @property
    def name(self):
        '''The :attr:`ceed.stage.StageShape.name` of the shape or shape group.
        '''
        return self.stage_shape.name

    def initialize_display(self, stage_shape, selection_controller):
        stage_shape.display = self
        self.stage_shape = stage_shape
        self.controller = self.selection_controller = selection_controller

        self.apply_kv()

    def remove_shape(self):
        self.stage_shape.stage.remove_shape(self.stage_shape)
        if self.selected:
            self.selection_controller.deselect_node(self)
        self.parent.remove_widget(self)

    @kv(proxy='app*')
    def apply_kv(self):
        app = _get_app()
        with KvContext():
            self.size_hint_y = None
            self.height = '34dp'
            self.size_hint_min_x @= self.minimum_width
            self.orientation = 'horizontal'
            self.use_parent = False
            self.spacing = '5dp'
            with self.canvas:
                color = Color()
                color.rgba ^= app.theme.primary_light if not self.selected else app.theme.primary
                rect = Rectangle()
                rect.size ^= self.size
                rect.pos ^= self.pos
            with Factory.FlatLabel(parent=self) as label:
                label.padding = '5dp', '5dp'
                label.flat_color @= app.theme.text_primary
                label.center_texture = False
                label.text @= self.stage_shape.name
                label.size_hint_min_x @= label.texture_size[0]
            with Factory.FlatImageButton(parent=self) as btn:
                btn.scale_down_color = True
                btn.source = 'flat_delete.png'
                btn.flat_color @= app.theme.accent
                with KvRule(btn.on_release, triggered_only=True):
                    self.remove_shape()


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
    by :meth:`ceed.view.controller.ViewControllerBase.get_all_shape_values`.
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

    def refresh_graph(self, stage, frame_rate):
        '''Re-samples the intensity values for the shapes from the stage.

        :Params:

            `stage`: :class:`ceed.stage.CeedStage`
                The stage from which to sample the shape intensity values.
            `frame_rate`: float
                The sampling rate used to sample the shape intensity values
                from the functions in time.

        '''
        frame_rate = self.frame_rate = float(frame_rate)
        vals = self.plot_values = App.get_running_app(
            ).view_controller.get_all_shape_values(stage, frame_rate)
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
        self.view_xmax = self.xmax = N / frame_rate

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
        depending on the state of the corresponding :attr:``r_selected.

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
            knspace.pinned_graph.add_widget(self)
            self.unpinned_root.dismiss()
        else:
            self.unpinned_parent.add_widget(self)
            self.unpinned_root.open()
