from kivy.event import EventDispatcher
from kivy_garden.graph import MeshLinePlot, Graph, LinePlot, ContourPlot, \
    PointsPlot
from kivy.properties import NumericProperty, ObjectProperty, DictProperty, \
    ReferenceListProperty, StringProperty, ListProperty, BooleanProperty
from kivy.utils import get_color_from_hex as rgb
from kivy.resources import resource_add_path
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.base import ExceptionManager, ExceptionHandler
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.modules import inspector
import math

from base_kivy_app.utils import ColorTheme
import base_kivy_app.app

import itertools
import sys
import numpy as np
from os.path import dirname, join, isdir
from collections import deque
from skimage import measure

resource_add_path(join(dirname(base_kivy_app.app.__file__), 'media'))
resource_add_path(join(dirname(base_kivy_app.app.__file__), 'media', 'flat_icons'))


class FormulaGraph(Graph):

    plot_widget = ObjectProperty(None)

    def __init__(self, **kwargs):
        self._with_stencilbuffer = False
        super(FormulaGraph, self).__init__(**kwargs)
        from kivy.core.window import Window
        Window.fbind('mouse_pos', self._update_mouse_pos)

    def _update_mouse_pos(self, instance, pos):
        plot_widget = self.plot_widget
        plot = plot_widget.plot
        if plot is None:
            return
        pos = self.to_widget(*pos, relative=True)

        if not self.collide_plot(*pos):
            return

        x, y = self.to_data(*pos)
        x2var = plot.x2_variable

        if not x2var:
            plot_widget.mouse_x_val = x
            plot_widget.mouse_y_val = y
            plot_widget.mouse_x2_val = None
        else:
            plot_widget.mouse_x_val = x
            plot_widget.mouse_x2_val = y
            if plot._yvals is None:
                plot_widget.mouse_y_val = 0
            else:
                n = plot.num_points
                xi = max(min(int(n * (x - plot.start) / (plot.end - plot.start)), n - 1), 0)
                yi = max(min(int(n * (y - plot.x2_start) / (plot.x2_end - plot.x2_start)), n - 1), 0)
                plot_widget.mouse_y_val = float(plot._yvals[xi, yi])

    def on_touch_down(self, touch):
        pos = self.to_local(*touch.pos, relative=True)
        if not self.collide_plot(*pos):
            return super(FormulaGraph, self).on_touch_down(touch)
        if super(FormulaGraph, self).on_touch_down(touch):
            return True

        if not touch.is_double_tap:
            return False

        x, y = self.to_data(*pos)
        plot = self.plot_widget.plot
        if plot is None:
            return False

        formula = plot.formula
        xvar = plot.x_variable
        x2var = plot.x2_variable

        if not x2var:
            if getattr(formula, '{}_src'.format(xvar), False):
                return False
            setattr(formula, xvar, x)
        else:
            if getattr(formula, '{}_src'.format(xvar), False) and \
                    getattr(formula, '{}_src'.format(x2var), False):
                return False

            if not getattr(formula, '{}_src'.format(xvar), False):
                setattr(formula, xvar, x)
            if not getattr(formula, '{}_src'.format(x2var), False):
                setattr(formula, x2var, y)
        return True


class FormulaPlot(EventDispatcher):

    graph = ObjectProperty(None, rebind=True)

    plot = ObjectProperty()

    start = NumericProperty(0)

    end = NumericProperty(100)

    x2_start = NumericProperty(0)

    x2_end = NumericProperty(100)

    y_start = NumericProperty(0)

    y_end = NumericProperty(100)

    num_points = NumericProperty(100)

    formula = ObjectProperty(None, rebind=True)

    x_variable = StringProperty('')

    x2_variable = StringProperty('')

    x_variable_formula = ObjectProperty(None)

    x2_variable_formula = ObjectProperty(None)

    y_variable = StringProperty('')

    num_contours = NumericProperty(5)

    track_ylim = BooleanProperty(False)

    _last_values = {}

    _yvals = None

    _y_range = None

    _contour_plots = []

    _value_plot = None

    colors = itertools.cycle([
        rgb('7dac9f'), rgb('dc7062'), rgb('66a8d4'), rgb('e5b060')])

    def __init__(self, **kwargs):
        super(FormulaPlot, self).__init__(**kwargs)
        self.fbind('start', self._update_from_params)
        self.fbind('end', self._update_from_params)
        self.fbind('x2_start', self._update_from_params)
        self.fbind('x2_end', self._update_from_params)
        self.fbind('num_points', self._update_from_params)
        self.fbind('x_variable', self._update_from_params)
        self.fbind('x2_variable', self._update_from_params)
        self.fbind('y_variable', self._update_from_params)
        self.fbind('formula', self._update_from_params)
        self.fbind('x_variable_formula', self._update_from_params)
        self.fbind('x2_variable_formula', self._update_from_params)

        self.fbind('y_start', self._update_y_vals)
        self.fbind('y_end', self._update_y_vals)
        self.fbind('num_contours', self._update_y_vals)

    def _update_from_params(self, *largs):
        self.refresh_plot(from_variables=False)

    def _update_y_vals(self, *largs):
        graph = self.graph
        if graph is None:
            return

        if self.x2_variable:
            if not self.plot or self._yvals is None:
                return
            self.plot.data = np.clip(self._yvals.T, self.y_start, self.y_end)
            self.compute_contours()
        else:
            graph.ymin = self.y_start
            graph.ymax = self.y_end
            graph.y_ticks_major = abs(graph.ymax - graph.ymin) / 4

    def compute_contours(self):
        graph = self.graph
        if graph is None or not self.x2_variable:
            return

        for plot in self._contour_plots:
            self.graph.remove_plot(plot)

        plots = self._contour_plots = []
        data = np.clip(self._yvals, self.y_start, self.y_end)
        xscale = (self.end - self.start) / self.num_points
        x2scale = (self.x2_end - self.x2_start) / self.num_points
        color = next(self.colors)

        for val in np.linspace(self.y_start, self.y_end, self.num_contours):
            contours = measure.find_contours(data, val)
            for contour in contours:
                contour[:, 0] *= xscale
                contour[:, 0] += self.start
                contour[:, 1] *= x2scale
                contour[:, 1] += self.x2_start

                plot = MeshLinePlot(color=color)
                plots.append(plot)
                graph.add_plot(plot)
                plot.points = contour

    def create_plot(self):
        x2 = self.x2_variable
        plot = self.plot
        if plot is not None:
            if (x2 and isinstance(plot, ContourPlot)
                    or not x2 and isinstance(plot, LinePlot)):
                return
            self.graph.remove_plot(plot)

        for plot in self._contour_plots:
            self.graph.remove_plot(plot)
            self._contour_plots = []

        self._yvals = None
        yvar = x2 or self.y_variable
        graph_theme = {
            'label_options': {
                'color': rgb('444444'),
                'bold': True},
            #'background_color': rgb('f8f8f2'),
            'tick_color': rgb('808080'),
            'border_color': rgb('808080'),
            'xlabel': '{} -- {}'.format(
                self.x_variable_formula.widget.name,
                self.formula.variable_descriptions.get(
                    self.x_variable, self.x_variable)),
            'ylabel': self.formula.variable_descriptions.get(
                yvar, yvar),
            'x_ticks_minor': 5,
            #'y_ticks_minor': 5,
            'y_grid_label': True,
            'x_grid_label': True,
            'padding': 5,
            'x_grid': True,
            'y_grid': True,
        }

        graph = self.graph
        for k, v in graph_theme.items():
            setattr(graph, k, v)

        if x2:
            self.plot = plot = ContourPlot(color=next(self.colors))
        else:
            self.plot = plot = LinePlot(color=next(self.colors), line_width=2)
        graph.add_plot(plot)

        self._value_plot = PointsPlot(color=next(self.colors), point_size=3)
        graph.add_plot(self._value_plot)

    def refresh_plot(self, from_variables=True):
        if self.graph is None or not self.x_variable or not self.y_variable:
            return
        self.create_plot()

        formula = self.formula
        xvar = self.x_variable
        x2var = self.x2_variable
        # new_vals = {
        #     var: getattr(formula, var) for var in formula.x_variables
        #     if var != xvar and var != x2var}

        # if from_variables and new_vals == self._last_values:
        #     return
        # self._last_values = new_vals

        start = self.start
        end = self.end
        n = self.num_points
        plot = self.plot
        graph = self.graph
        self._yvals = None

        if x2var:
            x2_start = self.x2_start
            x2_end = self.x2_end

            xvals = np.linspace(start, end, n)
            xvals = np.repeat(np.expand_dims(xvals, 1), n, 1)

            x2vals = np.linspace(x2_start, x2_end, n)
            x2vals = np.repeat(np.expand_dims(x2vals, 1), n, 1).T

            input_variables = [
                (self.x_variable_formula, xvar),
                (self.x2_variable_formula, x2var)]
            variables = {input_variables[0]: xvals, input_variables[1]: x2vals}

            yvals = formula.infer_variable_value(
                self.y_variable, variables, in_subtree={},
                input_variables=input_variables)
            if not isinstance(
                    yvals, np.ndarray) or n > 1 and yvals.shape != (n, n):
                yvals = np.zeros((n, n)) + float(yvals)
            else:
                yvals[np.logical_or(np.logical_or(
                    np.isnan(yvals), np.isinf(yvals)),
                    np.isneginf(yvals))] = -1000

            graph.xmin = start
            graph.xmax = end
            graph.ymin = x2_start
            graph.ymax = x2_end

            if self._y_range is None or self.track_ylim:
                self.y_start, self.y_end = self._y_range = \
                    float(np.min(yvals)), float(np.max(yvals))
            else:
                self._y_range = float(np.min(yvals)), float(np.max(yvals))

            graph.x_ticks_major = abs(end - start) / 10
            graph.y_ticks_major = abs(x2_end - x2_start) / 10
            graph.xlabel = '{} -- {}'.format(
                self.x_variable_formula.widget.name,
                self.formula.variable_descriptions.get(xvar, xvar))
            graph.ylabel = '{} -- {}'.format(
                self.x2_variable_formula.widget.name,
                self.formula.variable_descriptions.get(x2var, x2var))

            plot.xrange = (start, end)
            plot.yrange = (x2_start, x2_end)
            plot.data = np.clip(yvals.T, self.y_start, self.y_end)
            self._yvals = yvals

            self.compute_contours()
            self._value_plot.points = [(
                getattr(self.x_variable_formula, xvar),
                getattr(self.x2_variable_formula, x2var))]
        else:
            xvals = np.linspace(start, end, n)

            input_variables = [(self.x_variable_formula, xvar)]
            variables = {input_variables[0]: xvals}
            yvals = formula.infer_variable_value(
                self.y_variable, variables, in_subtree={},
                input_variables=input_variables)

            if not isinstance(
                    yvals, (np.ndarray, np.generic)) or n > 1 and len(yvals) == 1:
                yvals = np.zeros((n, )) + float(yvals)
            else:
                yvals[np.logical_or(np.logical_or(
                    np.isnan(yvals), np.isinf(yvals)),
                    np.isneginf(yvals))] = -1000

            ymin, ymax = np.min(yvals), np.max(yvals)
            if math.isclose(ymin, ymax):
                ydiff = abs(ymin) * 0.2
            else:
                ydiff = (ymax - ymin) * .02

            graph.xmin = start
            graph.xmax = end
            if self._y_range is None or self.track_ylim:
                self.y_start, self.y_end = self._y_range = \
                    float(ymin - ydiff), float(ymax + ydiff)
            else:
                self._y_range = float(ymin - ydiff), float(ymax + ydiff)
            graph.ymin = self.y_start
            graph.ymax = self.y_end

            graph.x_ticks_major = abs(end - start) / 10
            graph.y_ticks_major = abs(graph.ymax - graph.ymin) / 4
            graph.xlabel = '{} -- {}'.format(
                self.x_variable_formula.widget.name,
                self.formula.variable_descriptions.get(xvar, xvar))
            graph.ylabel = self.formula.variable_descriptions.get(
                self.y_variable, self.y_variable)

            plot.points = list(zip(xvals, yvals))
            self._value_plot.points = [(
                getattr(self.x_variable_formula, xvar),
                getattr(formula, self.y_variable))]

    def reset_y_axis(self):
        if not self.graph or not self.plot or self._y_range is None:
            return

        self.y_start, self.y_end = self._y_range


class CeedFormula(EventDispatcher):

    x_variables = ListProperty([])

    y_variables = ListProperty([])

    plots = ListProperty([])

    widget = ObjectProperty(None)

    variable_descriptions = DictProperty({})

    dependency_graph = {}
    '''Only y variables are listed here. '''

    y_dependency_ordered = []
    '''y variables ordered from leaf to dependant variables, such that
    any variable is only dependant on other y variables that are listed
    previously in `y_dependency_ordered`. '''

    def __init__(self, **kwargs):
        super(CeedFormula, self).__init__(**kwargs)

        for var in self.x_variables:
            self.fbind(var, self.update_result, var)
            var_src = '{}_src'.format(var)
            if not hasattr(self, var_src):
                continue
            self.bind_variable_to_src(var, var_src)

        yvars = self.y_variables
        deps = self.dependency_graph
        deps = {
            var: [v for v in dep_vars if v in yvars and v in deps]
            for var, dep_vars in deps.items() if var in yvars}
        y_ordered = self.y_dependency_ordered = [
            v for v in yvars if v not in deps]

        while deps:
            found = ''
            for var, dep_vars in deps.items():
                if not dep_vars:
                    y_ordered.append(var)
                    found = var
                    break

            if not found:
                raise Exception(
                    'Found y variables that depend on each other, so we cannot'
                    ' compute their dependency structure')

            deps = {
                var: [v for v in dep_vars if v != found]
                for var, dep_vars in deps.items() if var != found}

    def bind_variable_to_src(self, variable, variable_src):
        uid = [None, None, None]

        def watch_variable_src(*largs):
            if uid[0]:
                uid[1].unbind_uid(uid[2], uid[0])
                uid[0] = None

            src = getattr(self, variable_src)
            if not src:
                return

            obj, prop = src
            uid[1] = obj
            uid[2] = prop
            uid[0] = obj.fbind(prop, set_variable)
            setattr(self, variable, getattr(obj, prop))

        def set_variable(instance, value):
            setattr(self, variable, value)

        self.fbind(variable_src, watch_variable_src)
        watch_variable_src()

    def update_result(self, variable, *largs):
        for var in self.y_dependency_ordered:
            func = 'compute_{}'.format(var)
            setattr(self, var, getattr(self, func)())

        for plot in self.plots:
            plot.refresh_plot()

    def _get_src(self, variable):
        src = '{}_src'.format(variable)
        return getattr(self, src, None)

    def variables_in_subtree(self, variable, in_subtree, input_variables):
        '''We also check if the variable itself is a input variable. '''
        key = (self, variable)
        if key in in_subtree:
            return in_subtree[key]

        if key in input_variables:
            in_subtree[key] = True
            return True

        if variable in self.x_variables:
            var_src = self._get_src(variable)
            if not var_src:
                in_subtree[key] = False
                return False

            obj, prop = var_src
            in_subtree[key] = obj.variables_in_subtree(
                prop, in_subtree, input_variables)
            return in_subtree[key]

        assert variable in self.y_variables
        in_subtree[key] = any(
            self.variables_in_subtree(var, in_subtree, input_variables)
            for var in self.dependency_graph.get(variable, []))
        return in_subtree[key]

    def infer_variable_value(
            self, variable, variables, in_subtree, input_variables):
        '''Variables accumulates the values of veriables.
        in_subtree stores whether a variable contains any of the
        input_vars in it's dependency tree.'''
        key = (self, variable)
        if key in variables:
            # print('cached', id(self), variable, np.mean(variables[key]))
            return variables[key]

        if not self.variables_in_subtree(variable, in_subtree, input_variables):
            variables[key] = getattr(self, variable)
            # print('const', id(self), variable, np.mean(variables[key]))
            return variables[key]

        if variable in self.x_variables:
            assert key not in input_variables
            formula, prop = self._get_src(variable)
            variables[key] = formula.infer_variable_value(
                prop, variables, in_subtree, input_variables)
            # print('x_var', id(self), variable, np.mean(variables[key]))
            return variables[key]

        assert variable in self.y_variables
        for var in self.dependency_graph.get(variable, []):
            self.infer_variable_value(
                var, variables, in_subtree, input_variables)
        yfunc = getattr(self, 'compute_{}'.format(variable))
        variables[key] = yfunc(variables)
        # print('y_var', id(self), variable, np.mean(variables[key]))
        return variables[key]

    def get_variable_dep_leaves(self, variable):
        deps_graph = self.dependency_graph
        yvars = self.y_variables
        xvars = self.x_variables
        leaves = set()
        dep_x_vars = set()

        if variable in yvars:
            deps_vars = deque(deps_graph.get(variable, []))

            # go through all the deps of all yvars that depend on the variable
            while deps_vars:
                dep_var = deps_vars.popleft()
                if dep_var in xvars:
                    dep_x_vars.add(dep_var)
                else:
                    deps_vars.extend(deps_graph.get(dep_var, []))
        else:
            dep_x_vars.add(variable)

        for var in dep_x_vars:
            src = self._get_src(var)
            if not src:
                leaves.add((self, var))
            else:
                formula, prop = src
                leaves.update(formula.get_variable_dep_leaves(prop))
        return leaves


class LensFixedObjectFormula(CeedFormula):

    lens_pos_src = ObjectProperty(None)

    focal_length_src = ObjectProperty(None)

    object_pos_src = ObjectProperty(None)

    base_magnification_src = ObjectProperty(None)

    lens_pos = NumericProperty(0)

    focal_length = NumericProperty(0)

    object_pos = NumericProperty(0)

    base_magnification = NumericProperty(1)

    image_pos = NumericProperty(0)

    img_lens_pos = NumericProperty(0)

    magnification = NumericProperty(0)

    def __init__(self, **kwargs):
        self.x_variables.extend(
            ['focal_length', 'object_pos', 'lens_pos', 'base_magnification'])
        self.y_variables.extend(['image_pos', 'magnification', 'img_lens_pos'])
        self.dependency_graph = {
            'image_pos': ['lens_pos', 'object_pos', 'focal_length'],
            'magnification': ['lens_pos', 'object_pos', 'image_pos',
                              'base_magnification'],
            'img_lens_pos': ['image_pos', 'lens_pos']
        }
        super(LensFixedObjectFormula, self).__init__(**kwargs)

    def compute_image_pos(self, variables={}):
        lens_pos = variables.get((self, 'lens_pos'), self.lens_pos)
        object_pos = variables.get((self, 'object_pos'), self.object_pos)
        focal_length = variables.get((self, 'focal_length'), self.focal_length)
        object_dist = lens_pos - object_pos

        try:
            res = object_dist * focal_length / (
                    object_dist - focal_length) + lens_pos
        except ZeroDivisionError:
            res = -1000
        return res

    def compute_magnification(self, variables={}):
        lens_pos = variables.get((self, 'lens_pos'), self.lens_pos)
        object_pos = variables.get((self, 'object_pos'), self.object_pos)
        image_pos = variables.get((self, 'image_pos'), self.image_pos)
        base_mag = variables.get(
            (self, 'base_magnification'), self.base_magnification)
        object_dist = lens_pos - object_pos
        image_dist = image_pos - lens_pos

        try:
            res = -image_dist / object_dist * base_mag
        except ZeroDivisionError:
            res = -1000
        return res

    def compute_img_lens_pos(self, variables={}):
        image_pos = variables.get((self, 'image_pos'), self.image_pos)
        lens_pos = variables.get((self, 'lens_pos'), self.lens_pos)
        return image_pos - lens_pos


class LensFocalLengthFormula(CeedFormula):
    '''Only valid when not a virtual image.
    '''

    lens_pos_src = ObjectProperty(None)

    image_pos_src = ObjectProperty(None)

    object_pos_src = ObjectProperty(None)

    lens_pos = NumericProperty(0)

    image_pos = NumericProperty(0)

    object_pos = NumericProperty(0)

    focal_length = NumericProperty(0)

    magnification = NumericProperty(0)

    def __init__(self, **kwargs):
        self.x_variables.extend(['image_pos', 'object_pos', 'lens_pos'])
        self.y_variables.extend(['focal_length', 'magnification'])
        self.dependency_graph = {
            'focal_length': ['lens_pos', 'object_pos', 'image_pos'],
            'magnification': ['lens_pos', 'object_pos', 'image_pos']
        }
        super(LensFocalLengthFormula, self).__init__(**kwargs)

    def compute_focal_length(self, variables={}):
        lens_pos = variables.get((self, 'lens_pos'), self.lens_pos)
        object_pos = variables.get((self, 'object_pos'), self.object_pos)
        image_pos = variables.get((self, 'image_pos'), self.image_pos)
        object_dist = lens_pos - object_pos
        image_dist = image_pos - lens_pos

        try:
            res = 1 / (1 / image_dist + 1 / object_dist)
        except ZeroDivisionError:
            res = -1000
        return res

    def compute_magnification(self, variables={}):
        lens_pos = variables.get((self, 'lens_pos'), self.lens_pos)
        object_pos = variables.get((self, 'object_pos'), self.object_pos)
        image_pos = variables.get((self, 'image_pos'), self.image_pos)
        object_dist = lens_pos - object_pos
        image_dist = image_pos - lens_pos
        try:
            res = -image_dist / object_dist
        except ZeroDivisionError:
            res = -1000
        return res


class FormulaWidget(BoxLayout):

    formula = ObjectProperty(None)

    props_container_x = ObjectProperty(None)

    props_container_y = ObjectProperty(None)

    plots_container = ObjectProperty(None)

    description = StringProperty('')

    name = StringProperty('')

    hidden_variables = ObjectProperty(set(['base_magnification', ]))

    def populate_widget(self):
        props_container_x = self.props_container_x
        props_container_y = self.props_container_y
        formula = self.formula
        hidden_variables = self.hidden_variables

        def to_float(x):
            return float(x) if x else 0.
        to_str = lambda x: '{:0.4f}'.format(x) if x else '0'
        descriptions = self.formula.variable_descriptions

        def update(widget, src_name, *largs):
            widget.read_only = bool(getattr(formula, src_name))

        for var in sorted(formula.x_variables):
            if var in hidden_variables:
                continue

            display = Factory.VariableDisplay(
                name=descriptions.get(var, var),
                prop_from_display_setter=to_float,
                display_from_prop_setter=to_str,
                obj=formula,
                obj_prop=var)
            props_container_x.add_widget(display)

            src = '{}_src'.format(var)
            if hasattr(formula, src):
                formula.fbind(src, update, display, src)
                update(display, src)

        for var in sorted(formula.y_variables):
            if var in hidden_variables:
                continue

            display = Factory.VariableDisplay(
                name=descriptions.get(var, var),
                prop_from_display_setter=to_float,
                display_from_prop_setter=to_str,
                obj=formula,
                obj_prop=var,
                read_only=True)
            props_container_y.add_widget(display)

    def add_plot(self):
        plot = FormulaPlot(graph=None, formula=self.formula)
        self.formula.plots.append(plot)
        widget = PlotWidget(plot=plot, formula_widget=self)
        self.plots_container.add_widget(widget)
        plot.refresh_plot(from_variables=False)
        widget.populate_x_variables()

    def remove_plot(self, plot_widget):
        self.formula.plots.remove(plot_widget.plot)
        self.plots_container.remove_widget(plot_widget.__self__)


class PropertyDisplayBinding(EventDispatcher):

    prop_from_display_setter = ObjectProperty(lambda x: x)

    display_from_prop_setter = ObjectProperty(lambda x: x)

    obj = ObjectProperty(None)

    obj_prop = StringProperty('')

    read_only = BooleanProperty(False)

    prop_value = ObjectProperty('')

    def __init__(self, **kwargs):
        super(PropertyDisplayBinding, self).__init__(**kwargs)
        uid = [None, None, None]

        def watch_prop():
            if uid[0]:
                uid[1].unbind_uid(uid[2], uid[0])
                uid[0] = None

            if not self.obj or not self.obj_prop:
                return

            uid[1] = obj = self.obj
            uid[2] = prop = self.obj_prop
            uid[0] = obj.fbind(prop, set_display)
            self.prop_value = self.display_from_prop_setter(getattr(obj, prop))

        def set_display(instance, value):
            self.prop_value = self.display_from_prop_setter(value)

        self.fbind('obj', watch_prop)
        self.fbind('obj_prop', watch_prop)
        watch_prop()

    def set_obj_property(self, value):
        if not self.obj or not self.obj_prop or self.read_only:
            return

        setattr(self.obj, self.obj_prop, self.prop_from_display_setter(value))


class FormulaVariableBehavior(EventDispatcher):

    name = StringProperty('')


class PlotWidget(BoxLayout):

    plot = ObjectProperty(None, rebind=True)

    formula_widget = ObjectProperty(None)

    mouse_x_val = NumericProperty(0)

    mouse_x2_val = NumericProperty(None, allownone=True)

    mouse_y_val = NumericProperty(0)

    graph_min_height = NumericProperty(200)

    _names_to_x_variables = DictProperty({})

    def select_x_variable(self, x_prop, variable_name):
        formula_prop = '{}_variable_formula'.format(x_prop)
        variable_prop = '{}_variable'.format(x_prop)

        if not variable_name:
            setattr(self.plot, formula_prop, self.plot.formula)
            setattr(self.plot, variable_prop, '')
        else:
            formula, var = self._names_to_x_variables[variable_name]
            setattr(self.plot, formula_prop, formula)
            setattr(self.plot, variable_prop, var)

    def populate_x_variables(self):
        formula = self.formula_widget.formula
        deps = set()
        for var in formula.y_variables:
            deps.update(formula.get_variable_dep_leaves(var))

        self._names_to_x_variables = {
            '{}: {}'.format(f.widget.name, var): (f, var) for (f, var) in deps
            if var not in f.widget.hidden_variables
        }


Factory.register('PropertyDisplayBinding', cls=PropertyDisplayBinding)
Factory.register('FormulaVariableBehavior', cls=FormulaVariableBehavior)


class OpticsApp(App):

    theme = ObjectProperty(None, rebind=True)

    formula_container_widget = ObjectProperty(None)

    focal_len_from_io_f = ObjectProperty(None)

    image_from_f = ObjectProperty(None)

    objective_lens = ObjectProperty(None)

    cam_lens_further = ObjectProperty(None)

    cam_lens_closer = ObjectProperty(None)

    cam_lens_closer2 = ObjectProperty(None)

    def __init__(self, **kwargs):
        self.theme = ColorTheme()
        super(OpticsApp, self).__init__(**kwargs)
        self.focal_len_from_io_f = LensFocalLengthFormula()
        self.image_from_f = LensFixedObjectFormula()

        self.objective_lens = LensFixedObjectFormula()
        self.cam_lens_further = LensFixedObjectFormula()
        self.cam_lens_closer = LensFixedObjectFormula()
        self.cam_lens_closer2 = LensFixedObjectFormula()

        self.cam_lens_further.object_pos_src = self.objective_lens, 'image_pos'
        self.cam_lens_closer.object_pos_src = self.cam_lens_further, 'image_pos'
        self.cam_lens_closer2.object_pos_src = self.cam_lens_closer, 'image_pos'

        self.cam_lens_further.base_magnification_src = (
            self.objective_lens, 'magnification')
        self.cam_lens_closer.base_magnification_src = (
            self.cam_lens_further, 'magnification')
        self.cam_lens_closer2.base_magnification_src = (
            self.cam_lens_closer, 'magnification')

    def build(self):
        root = Factory.FormulaRoot()
        container = self.formula_container_widget = root.children[0].children[0]

        self.focal_len_from_io_f.widget = widget = FormulaWidget(
            formula=self.focal_len_from_io_f,
            description='Compute f from object/image distance',
            name='Lf')
        widget.populate_widget()
        container.add_widget(widget)

        self.image_from_f.widget = widget = FormulaWidget(
            formula=self.image_from_f,
            description='Compute image from fixed f/object distance',
            name='Li')
        widget.populate_widget()
        container.add_widget(widget)

        self.objective_lens.widget = widget = FormulaWidget(
            formula=self.objective_lens,
            description='(1/4) First lens in the sequence.',
            name='L1')
        widget.populate_widget()
        container.add_widget(widget)

        self.cam_lens_further.widget = widget = FormulaWidget(
            formula=self.cam_lens_further,
            description='(2/4) Second lens in the sequence.',
            name='L2')
        widget.populate_widget()
        container.add_widget(widget)

        self.cam_lens_closer.widget = widget = FormulaWidget(
            formula=self.cam_lens_closer,
            description='(3/4) Third lens in the sequence.',
            name='L3')
        widget.populate_widget()
        container.add_widget(widget)

        self.cam_lens_closer2.widget = widget = FormulaWidget(
            formula=self.cam_lens_closer2,
            description='((4/4) Fourth lens in the sequence.',
            name='L4')
        widget.populate_widget()
        container.add_widget(widget)

        from kivy.core.window import Window
        inspector.create_inspector(Window, root)
        return root


if __name__ == '__main__':
    class _AppHandler(ExceptionHandler):

        def handle_exception(self, inst):
            Logger.error(inst, exc_info=sys.exc_info())
            return ExceptionManager.PASS

    handler = _AppHandler()
    ExceptionManager.add_handler(handler)

    app = OpticsApp()
    try:
        app.run()
    except Exception as e:
        Logger.error(e, exc_info=sys.exc_info())

    ExceptionManager.remove_handler(handler)
