from copy import deepcopy
import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty, OptionProperty
from kivy.core.window import Window

from ceed.utils import WidgetList, ShowMoreSelection, BoxSelector, \
    ShowMoreBehavior, fix_name, ColorBackgroundBehavior
from ceed.stage import StageFactory, CeedStage
from ceed.function.func_widgets import FuncWidget, FuncWidgetGroup, \
    FunctionFactory


class StageList(ShowMoreSelection, WidgetList, BoxLayout):

    def __init__(self, **kwargs):
        super(StageList, self).__init__(**kwargs)
        self.nodes_order_reversed = False

    def add_func(self, name):
        after = None
        if not self.selected_nodes:
            return

        src_func = FunctionFactory.avail_funcs[name]
        widget = self.selected_nodes[0]
        if isinstance(widget, StageWidget):
            parent = widget.stage
        elif isinstance(widget, FuncWidgetGroup):
            parent = widget.func
            if parent.parent_in_other_children(src_func):
                return
        elif isinstance(widget, FuncWidget):
            after = widget.func
            if after.parent_func:
                parent = after.parent_func
                if parent.parent_in_other_children(src_func):
                    return
            else:
                parent = widget.func_controller

        parent.add_func(deepcopy(src_func), after=after)

    def get_selected_shape_stage(self):
        if self.selected_nodes:
            widget = self.selected_nodes[0]
            if isinstance(widget, StageWidget):
                return widget.stage
            if isinstance(widget, StageShapeDisplay):
                return widget.stage_shape.stage

    def add_shapes(self, shapes):
        stage = self.get_selected_shape_stage()
        if not stage:
            return
        for shape in shapes:
            stage.add_shape(shape)

    def add_selected_shapes(self):
        self.add_shapes(knspace.painter.selected_shapes)

    def add_selected_shape_groups(self):
        self.add_shapes(knspace.painter.selected_groups)

    def add_shape_by_name(self, name):
        if name in knspace.painter.shape_names:
            self.add_shapes([knspace.painter.shape_names[name]])
        elif name in knspace.painter.shape_group_names:
            self.add_shapes([knspace.painter.shape_group_names[name]])

    def add_stage(self):
        parent = None
        if self.selected_nodes:
            widget = self.selected_nodes[0]
            if not isinstance(widget, StageWidget):
                return
            parent = widget.stage
        else:
            parent = StageFactory

        parent.add_stage(CeedStage())

    def get_selectable_nodes(self):
        return [d for stage in StageFactory.stages
                for d in stage.display.get_visible_children()]


class StageWidget(ShowMoreBehavior, ColorBackgroundBehavior, BoxLayout):

    stage = ObjectProperty(None, rebind=True)

    selected = BooleanProperty(False)

    stage_widget = ObjectProperty(None)

    func_widget = ObjectProperty(None)

    shape_widget = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(StageWidget, self).__init__(**kwargs)
        self.settings.parent.remove_widget(self.settings)

    @property
    def name(self):
        return self.stage.name

    def set_func_controller(self, func_widget):
        for func in func_widget.func.get_funcs():
            func.display.func_controller = self.stage
            func.display.display_parent = self.func_widget
            func.display.selection_controller = knspace.stages

    def remove_from_parent(self):
        if self.stage.parent_stage:
            self.stage.parent_stage.remove_stage(self.stage)
        else:
            StageFactory.remove_stage(self.stage)

    def show_stage(self):
        parent = self.stage.parent_stage
        if parent:
            i = len(parent.stages) - parent.stages.index(self.stage) - 1
            parent.display.stage_widget.add_widget(self, index=i)
            if self.ids.name_input in self.settings.children:
                self.settings.remove_widget(self.ids.name_input)
        else:
            knspace.stages.add_widget(self)

    def hide_stage(self):
        for child in self.get_visible_children():
            if child.selected:
                knspace.stages.deselect_node(child)
                break

        if self.parent:
            self.parent.remove_widget(self)

    def get_visible_children(self):
        yield self
        for stage in self.stage.stages:
            for child in stage.display.get_visible_children():
                yield child

        for func in self.stage.functions:
            for f in func.get_funcs():
                yield f.display

        for shape in self.stage.shapes:
            yield shape.display


class StageShapeDisplay(ColorBackgroundBehavior, BoxSelector):

    stage_shape = ObjectProperty(None, rebind=True)

    selected = BooleanProperty(False)

    @property
    def name(self):
        return self.stage_shape.name

    def show_widget(self):
        stage = self.stage_shape.stage
        i = len(stage.shapes) - stage.shapes.index(self.stage_shape) - 1
        stage.display.shape_widget.add_widget(self, index=i)

    def hide_widget(self):
        if self.selected:
            knspace.stages.deselect_node(self)

        if self.parent:
            self.parent.remove_widget(self)
