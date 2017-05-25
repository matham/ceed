'''Stage widgets
===================

Defines the GUI components used with :mod:`ceed.stage`.
'''
from copy import deepcopy
from scipy.signal import decimate
import numpy as np

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty, OptionProperty
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.garden.graph import MeshLinePlot
from kivy.metrics import dp
from kivy.app import App
from kivy.utils import get_color_from_hex
from kivy.graphics import Rectangle, Color, Line

from ceed.utils import fix_name
from ceed.graphics import WidgetList, ShowMoreSelection, BoxSelector, \
    ShowMoreBehavior, ColorBackgroundBehavior
from ceed.stage import StageFactory, CeedStage
from ceed.function.func_widgets import FuncWidget, FuncWidgetGroup, \
    FunctionFactory
from ceed.view.controller import ViewController
from ceed.shape import get_painter

__all__ = ('StageList', 'StageWidget', 'StageShapeDisplay', 'ShapePlot',
           'StageGraph')


class StageList(ShowMoreSelection, WidgetList, BoxLayout):
    '''Widget that shows the list of all the stages.
    '''

    def __init__(self, **kwargs):
        super(StageList, self).__init__(**kwargs)
        self.nodes_order_reversed = False

    def add_func(self, name):
        '''Adds a copy of the function to the stage at the currently selected
        stage.

        :Params:

            `name`: str
                The name of the function instance from the
                :attr:`ceed.function.FunctionFactory` to use.
        '''
        after = None
        if not self.selected_nodes:
            return

        src_func = FunctionFactory.funcs_inst[name]
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
        '''Returns the :class:`ceed.stage.CeedStage` instance a currently
        selected in the GUI.
        '''
        if self.selected_nodes:
            widget = self.selected_nodes[0]
            if isinstance(widget, StageWidget):
                return widget.stage
            if isinstance(widget, StageShapeDisplay):
                return widget.stage_shape.stage

    def add_shapes(self, shapes):
        '''Adds the shapes to the currently selected
        :class:`ceed.stage.CeedStage`.

        :Params:

            `shapes`: iterable
                A list of :class:`ceed.shape.CeedShape` or
                :class:`ceed.shape.CeedShapeGroup` instances to add.
        '''
        stage = self.get_selected_shape_stage()
        if not stage:
            return
        for shape in shapes:
            stage.add_shape(shape)

    def add_selected_shapes(self):
        '''Adds the currently selected :class:`ceed.shape.CeedShape` instances
        to the currently selected :class:`ceed.stage.CeedStage`.
        '''
        self.add_shapes(knspace.painter.selected_shapes)

    def add_selected_shape_groups(self):
        '''Adds the currently selected :class:`ceed.shape.CeedShapeGroup`
        instances to the currently selected :class:`ceed.stage.CeedStage`.
        '''
        self.add_shapes(knspace.painter.selected_groups)

    def add_shape_by_name(self, name):
        '''Adds the :class:`ceed.shape.CeedShape` or
        :class:`ceed.shape.CeedShapeGroup` with name ``name`` to the currently
        selected :class:`ceed.stage.CeedStage`.
        '''
        if name in knspace.painter.shape_names:
            self.add_shapes([knspace.painter.shape_names[name]])
        elif name in knspace.painter.shape_group_names:
            self.add_shapes([knspace.painter.shape_group_names[name]])

    def add_stage(self):
        '''Adds a new :class:`ceed.stage.CeedStage` instance to the currently
        selected :class:`ceed.stage.CeedStage` instance or to the root list
        (:attr:`ceed.stage.StageFactory`).
        '''
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


class StageWidget(ShowMoreBehavior, BoxLayout):
    '''The widget displayed for an :class:`ceed.stage.CeedStage` instance.
    '''

    stage = ObjectProperty(None, rebind=True)
    '''The :class:`ceed.stage.CeedStage` instance attached to the widget.
    '''

    selected = BooleanProperty(False)

    stage_widget = ObjectProperty(None)
    '''The internal widget container to which children
    :class:`StageWidget` widget instances are added.
    '''

    func_widget = ObjectProperty(None)
    '''The internal widget container to which children
    :class:`ceed.func_widgets.FuncWidget` or
    :class:`ceed.func_widgets.FuncWidgetGroup` widget instances are added.
    '''

    shape_widget = ObjectProperty(None)
    '''The internal widget container to which children
    :class:`StageShapeDisplay` widget instances are added.
    '''

    def __init__(self, **kwargs):
        super(StageWidget, self).__init__(**kwargs)
        self.settings_root.parent.remove_widget(self.settings_root)

    @property
    def name(self):
        '''The :attr:`ceed.stage.CeedStage.name` of the stage.
        '''
        return self.stage.name

    def set_func_controller(self, func_widget):
        '''Sets the controller attributes for the
        :class:`ceed.func_widgets.FuncWidget` and
        :class:`ceed.func_widgets.FuncWidgetGroup` widgets for all the
        function children of the ``func_widget`` instance (which is
        also a :class:`ceed.func_widgets.FuncWidget` or
        :class:`ceed.func_widgets.FuncWidgetGroup`).
        '''
        for func in func_widget.func.get_funcs():
            func.display.func_controller = self.stage
            func.display.display_parent = self.func_widget
            func.display.selection_controller = knspace.stages

    def remove_from_parent(self):
        '''Removes the stage from the parent stage or from the global
        :attr:`ceed.stage.StageFactory` if it doesn't have a parent.
        '''
        if self.stage.parent_stage:
            self.stage.parent_stage.remove_stage(self.stage)
        else:
            StageFactory.remove_stage(self.stage)

    def show_stage(self):
        '''Adds the stage's widget in the GUI.
        '''
        parent = self.stage.parent_stage
        if parent:
            i = len(parent.stages) - parent.stages.index(self.stage) - 1
            parent.display.stage_widget.add_widget(self, index=i)
            if self.ids.name_input in self.settings.children:
                self.settings.remove_widget(self.ids.name_input)
        else:
            knspace.stages.add_widget(self)

    def hide_stage(self):
        '''Removes the stage's widget in the GUI.
        '''
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


class StageShapeDisplay(BoxSelector):
    '''The widget used for the :class:`ceed.stage.StageShape`.
    '''

    stage_shape = ObjectProperty(None, rebind=True)
    '''The :class:`ceed.stage.StageShape` instance that this widget displays.
    '''

    selected = BooleanProperty(False)

    @property
    def name(self):
        '''The :attr:`ceed.stage.StageShape.name` of the shape or shape group.
        '''
        return self.stage_shape.name

    def show_widget(self):
        '''Adds the shapes's widget in the GUI.
        '''
        stage = self.stage_shape.stage
        i = len(stage.shapes) - stage.shapes.index(self.stage_shape) - 1
        stage.display.shape_widget.add_widget(self, index=i)

    def hide_widget(self):
        '''Removes the shapes's widget from the GUI.
        '''
        if self.selected:
            knspace.stages.deselect_node(self)

        if self.parent:
            self.parent.remove_widget(self)


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
        b = self.r_btn = Factory.ShapeGraphSelector(controller=graph, text='R')
        b.controller = graph
        b = self.g_btn = Factory.ShapeGraphSelector(controller=graph, text='G')
        b.controller = graph
        b = self.b_btn = Factory.ShapeGraphSelector(controller=graph, text='B')
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
        vals = self.plot_values = ViewController.get_all_shape_values(
            stage, frame_rate)
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
