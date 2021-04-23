"""Optics App
==============

App used to help design the lens system used in the microscope. See
:ref:`microscope-optics` for the full details of how the app is used to design
and select all the lenses in the system.

There are three model sections in the app:

Focal length / mag from position
--------------------------------

This section models how the focal length and mag changes given the positions
of the lens, the object seen by the lens, and the image of the object
on the other side of the lens.

You can also add graphs to this section to visualize in 2D and 3D the effect of
changing these parameters.

Image/lens pos and mag from lens/object pos and focal length
-------------------------------------------------------------

This section models how the image and les pos need to be given a desired
lens and object position and focal length.

You can also add graphs to this section to visualize in 2D and 3D the effect of
changing these parameters.

4-lens system
-------------

This section lets you chain upto 4 lenses in a system and for each lens it lets
you change its focal length, its position, and the object position and it
estimates the image position and the mag for that and all subsequent lenses
based on the previous lenses in the chain.

The object position can only be given for the first lens, the others are
computed from the previous lenses.

For each lens in the chain you can also add 2D and 3D graphs that lets you
explore how the output parameters may change, given input parameters.
"""
from kivy.event import EventDispatcher
from kivy_garden.graph import MeshLinePlot, Graph, LinePlot, ContourPlot, \
    PointPlot
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
from typing import List, Dict, Tuple, Optional
import numpy as np
from os.path import dirname, join, isdir
from collections import deque
from skimage import measure

__all__ = (
    'OpticsApp', 'PlotWidget', 'FormulaVariableBehavior',
    'PropertyDisplayBinding', 'FormulaWidget', 'LensFocalLengthFormula',
    'LensFixedObjectFormula', 'CeedFormula', 'FormulaPlot', 'FormulaGraph')

resource_add_path(join(dirname(base_kivy_app.app.__file__), 'media'))
resource_add_path(
    join(dirname(base_kivy_app.app.__file__), 'media', 'flat_icons'))


class FormulaGraph(Graph):
    """The graph on which the 2D/3D plots are drawn.
    """

    plot_widget: 'PlotWidget' = ObjectProperty(None)
    """The :class:`PlotWidget` that contains this graph.
    """

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
                xi = max(min(
                    int(n * (x - plot.start) / (plot.end - plot.start)),
                    n - 1), 0)
                yi = max(min(
                    int(n * (y - plot.x2_start) /
                        (plot.x2_end - plot.x2_start)),
                    n - 1), 0)
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
    """Given a :class:`CeedFormula`, it uses that to compute the the 2D/3D
    formula output, given the range of input values.

    It supports updating the plot from a formula whose input is the output of
    another formula in a chain. In which case it starts with the input to root
    formula and evaluates all the formula in the chain until the last one. Then
    uses that to evaluate the final formula on a range of values that is then
    displayed in the graph.
    """

    graph: 'FormulaGraph' = ObjectProperty(None, rebind=True)
    """The graph in which to display the formula outputs.
    """

    plot = ObjectProperty()
    """The specific 2D/3D plot in the graph.
    """

    start = NumericProperty(0)
    """The horizontal axis start value from which to evaluate.
    """

    end = NumericProperty(100)
    """The horizontal axis end value until which to evaluate.
    """

    x2_start = NumericProperty(0)
    """The vertical axis start value from which to evaluate/display.
    """

    x2_end = NumericProperty(100)
    """The vertical axis end value until which to evaluate/display.
    """

    y_start = NumericProperty(0)
    """The depth (z) axis start value from which to display, if 3D.
    """

    y_end = NumericProperty(100)
    """The depth (z) axis end value until which to display, if 3D.
    """

    num_points = NumericProperty(100)
    """The number of horizontal/vertical points at which to evaluate the
    formula.
    """

    formula: 'CeedFormula' = ObjectProperty(None, rebind=True)
    """The :class:`CeedFormula` that takes the x (and x2 if 3D) input values and
    computes the output values.
    """

    x_variable = StringProperty('')
    """Name of the horizontal input variable.
    """

    x2_variable = StringProperty('')
    """Name of the vertical input variable, if any (3D).
    """

    x_variable_formula: 'CeedFormula' = ObjectProperty(None)
    """The formula used to generate the horizontal input variable, if it is
    part of a lens chain and the variable is the result of a previous
    formula in the chain.
    """

    x2_variable_formula: 'CeedFormula' = ObjectProperty(None)
    """The formula used to generate the vertical input variable, if it is
    part of a lens chain and the variable is the result of a previous
    formula in the chain.
    """

    y_variable = StringProperty('')
    """The name of the output (y) variable computed.
    """

    num_contours = NumericProperty(5)
    """The number of equal value contours to display for 3D plots.
    """

    track_ylim = BooleanProperty(False)
    """For 3D plots, whether to update the plot outout range as the output (y)
    range changes or if the leave it unchanged.
    """

    _last_values = {}

    _yvals = None

    _y_range = None

    _contour_plots = []

    _value_plot = None

    colors = itertools.cycle([
        rgb('7dac9f'), rgb('dc7062'), rgb('66a8d4'), rgb('e5b060')])
    """Plot colors to use.
    """

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
        """Computes the equal value contours for 3D plots.
        """
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
        """Creates and displays the plot for this formula/graph.
        """
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
            # 'background_color': rgb('f8f8f2'),
            'tick_color': rgb('808080'),
            'border_color': rgb('808080'),
            'xlabel': '{} -- {}'.format(
                self.x_variable_formula.widget.name,
                self.formula.variable_descriptions.get(
                    self.x_variable, self.x_variable)),
            'ylabel': self.formula.variable_descriptions.get(
                yvar, yvar),
            'x_ticks_minor': 5,
            # 'y_ticks_minor': 5,
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

        self._value_plot = PointPlot(color=next(self.colors), point_size=3)
        graph.add_plot(self._value_plot)

    def refresh_plot(self, from_variables=True):
        """Updates plot when any of the variables or parameters change.
        """
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
                    yvals,
                    (np.ndarray, np.generic)) or n > 1 and len(yvals) == 1:
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
        """Resets the y (output)-axis to the previous value.
        """
        if not self.graph or not self.plot or self._y_range is None:
            return

        self.y_start, self.y_end = self._y_range


class CeedFormula(EventDispatcher):
    """A formula that computes a output value(s), given some input value
    or input value range.

    The input value(s) can be formula themselves so that we can have a formula
    DAG, with some formula leaves. This allows the output of a leaf to be
    re-computed when any of the inputs or its parents change in the graph.

    The formula is computed from properties that are input parameters and it
    generates one or more output values that are also stored as properties.

    The input properties (e.g. ``lens_pos``) can have a second, corresponding
    property with the ``_src`` suffix (e.g. ``lens_pos_src``), which if set to
    a tuple of (:class:`CeedFormula`, property_name), the given property of the
    formula will be used for the input value (``lens_pos_src``) instead of the
    property of this instance (``lens_pos``). This allows chaining.

    Additionally, each output property must have a corresponding method with
    a ``compute_`` prefix that when called returns the output given the
    inputs stored in the class properties. It can be given a set of
    :class:`CeedFormula` that is used to look up values for any of the input
    properties, instead of using the value currently stored in the instance
    property for that input.
    """

    x_variables: List[str] = ListProperty([])
    """List of properties of this class (by name) whose values are used as
    inputs to the formula's function.

    There must be a corresponding method with a ``compute_`` prefix for each
    variable.

    There may also be a corresponding property with a `_src`` suffix that
    contains a formula to use to look up the property value as the output of
    that formula, instead of using this property value directly.
    """

    y_variables: List[str] = ListProperty([])
    """List of properties of this class (by name) whose values are outputs of
    the formula's function.

    Items in :attr:`y_variables` may depend on other items in
    :attr:`y_variables` as input to their calculation. See
    :attr:`y_dependency_ordered`.
    """

    plots = ListProperty([])
    """All the plots that use this formula and need to be updated when the
    inputs change.
    """

    widget = ObjectProperty(None)

    variable_descriptions: Dict[str, str] = DictProperty({})
    """A mapping that gets a nicer description for each variable.
    """

    dependency_graph: Dict[str, List[str]] = {}
    '''Mapping that maps each y-variable to a list of all the y and x variables
    it depends on.
    '''

    y_dependency_ordered: List[str] = []
    '''As mentioned, y (output) variables may depend on other output variables.
    This orders y variables from leaf to dependant variables, such that
    any variable is only dependant on other y variables that are listed
    previously in :attr:`y_dependency_ordered`.

    It is automatically computed.
    '''

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
        """For each x-variable, if there's a corresponding ``_src`` suffixed
        variable that is set, this method will track that formula and update the
        our property when the source changes.
        """
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
        """Called automatically when a x-variable changes and it recomputes
        all the y-variables.
        """
        for var in self.y_dependency_ordered:
            func = 'compute_{}'.format(var)
            setattr(self, var, getattr(self, func)())

        for plot in self.plots:
            plot.refresh_plot()

    def _get_src(self, variable):
        src = '{}_src'.format(variable)
        return getattr(self, src, None)

    def variables_in_subtree(self, variable, in_subtree, input_variables):
        '''Checks whether the variable or its dependencies is a input variable.
        If so it'll need to be computed over the range that the input is
        sampled on. If not, it's a constant value that we can just get from
        the formula properties.

        We also check if the variable itself is a input variable.
        '''
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
        '''Computes the value of variable for the range of values of the
        input variables.

        Variables accumulates the values of variables as they are computed in
        the graph, starting from the root.

        in_subtree stores whether a variable contains any of the
        input_variables in it's dependency tree (it depends on it).
        '''
        key = (self, variable)
        if key in variables:
            return variables[key]

        if not self.variables_in_subtree(variable, in_subtree, input_variables):
            variables[key] = getattr(self, variable)
            return variables[key]

        if variable in self.x_variables:
            assert key not in input_variables
            formula, prop = self._get_src(variable)
            variables[key] = formula.infer_variable_value(
                prop, variables, in_subtree, input_variables)
            return variables[key]

        assert variable in self.y_variables
        for var in self.dependency_graph.get(variable, []):
            self.infer_variable_value(
                var, variables, in_subtree, input_variables)
        yfunc = getattr(self, 'compute_{}'.format(variable))
        variables[key] = yfunc(variables)
        return variables[key]

    def get_variable_dep_leaves(self, variable):
        """Gets set of all the (formula, variable) tuples that this variable
        depends on upto the root, but only those that are roots in the sense
        that the variables don't depend on other variable themselves.

        This will let us start from them and compute the full graph until
        reaching the the variable.
        """
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
    """Computes the properties of a lens, where the object (input distance)
    is fixed.
    """

    lens_pos_src: Optional[CeedFormula] = ObjectProperty(None)
    """The previous formula in the chain, if any, that computes this input
    variable as its output.
    """

    focal_length_src: Optional[CeedFormula] = ObjectProperty(None)
    """The previous formula in the chain, if any, that computes this input
    variable as its output.
    """

    object_pos_src: Optional[CeedFormula] = ObjectProperty(None)
    """The previous formula in the chain, if any, that computes this input
    variable as its output.
    """

    base_magnification_src: Optional[CeedFormula] = ObjectProperty(None)
    """The previous formula in the chain, if any, that computes this input
    variable as its output.
    """

    lens_pos = NumericProperty(0)
    """The value of the input variable (automatically set from ``_src`` if set).
    """

    focal_length = NumericProperty(0)
    """The value of the input variable (automatically set from ``_src`` if set).
    """

    object_pos = NumericProperty(0)
    """The value of the input variable (automatically set from ``_src`` if set).
    """

    base_magnification = NumericProperty(1)
    """The value of the input variable (automatically set from ``_src`` if set).
    """

    image_pos = NumericProperty(0)
    """The computed output variable value.
    """

    img_lens_pos = NumericProperty(0)
    """The computed output variable value.
    """

    magnification = NumericProperty(0)
    """The computed output variable value.
    """

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
        """Computes and returns this output variable.

        If the ``variables`` dict contains a formula for a input variable, that
        variable value is gotten from from ``variables``. Otherwise it gets it
        from the variable property. THis allows us to support computing
        whole ranges for the input variables as their values can be an array.
        """
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
        """Similar to :meth:`compute_image_pos`.
        """
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
        """Similar to :meth:`compute_image_pos`.
        """
        image_pos = variables.get((self, 'image_pos'), self.image_pos)
        lens_pos = variables.get((self, 'lens_pos'), self.lens_pos)
        return image_pos - lens_pos


class LensFocalLengthFormula(CeedFormula):
    '''Only valid when not a virtual image.
    '''

    lens_pos_src: Optional[CeedFormula] = ObjectProperty(None)
    """The previous formula in the chain, if any, that computes this input
    variable as its output.
    """

    image_pos_src: Optional[CeedFormula] = ObjectProperty(None)
    """The previous formula in the chain, if any, that computes this input
    variable as its output.
    """

    object_pos_src: Optional[CeedFormula] = ObjectProperty(None)
    """The previous formula in the chain, if any, that computes this input
    variable as its output.
    """

    lens_pos = NumericProperty(0)
    """The value of the input variable (automatically set from ``_src`` if set).
    """

    image_pos = NumericProperty(0)
    """The value of the input variable (automatically set from ``_src`` if set).
    """

    object_pos = NumericProperty(0)
    """The value of the input variable (automatically set from ``_src`` if set).
    """

    focal_length = NumericProperty(0)
    """The computed output variable value.
    """

    magnification = NumericProperty(0)
    """The computed output variable value.
    """

    def __init__(self, **kwargs):
        self.x_variables.extend(['image_pos', 'object_pos', 'lens_pos'])
        self.y_variables.extend(['focal_length', 'magnification'])
        self.dependency_graph = {
            'focal_length': ['lens_pos', 'object_pos', 'image_pos'],
            'magnification': ['lens_pos', 'object_pos', 'image_pos']
        }
        super(LensFocalLengthFormula, self).__init__(**kwargs)

    def compute_focal_length(self, variables={}):
        """Similar to :meth:`LensFixedObjectFormula.compute_image_pos`.
        """
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
        """Similar to :meth:`LensFixedObjectFormula.compute_image_pos`.
        """
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
    """Widget that displays a formula, its inputs, outputs and graphs.
    """

    formula: CeedFormula = ObjectProperty(None)
    """The formula visulized by the widget.
    """

    props_container_x = ObjectProperty(None)
    """Widget container for all the input values.
    """

    props_container_y = ObjectProperty(None)
    """Widget container for all the output values.
    """

    plots_container = ObjectProperty(None)
    """Widget container for all the graphs displayed for the formula.
    """

    description = StringProperty('')
    """Description shown for the formula.
    """

    name = StringProperty('')
    """Name shown for the formula.
    """

    hidden_variables = ObjectProperty({'base_magnification'})
    """List of all the input/output variables that are not shown in the GUI.
    """

    def populate_widget(self):
        """Adds widgets for all the variables.
        """
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
        """Adds a new plot for the formula.
        """
        plot = FormulaPlot(graph=None, formula=self.formula)
        self.formula.plots.append(plot)
        widget = PlotWidget(plot=plot, formula_widget=self)
        self.plots_container.add_widget(widget)
        plot.refresh_plot(from_variables=False)
        widget.populate_x_variables()

    def remove_plot(self, plot_widget):
        """Removes an existing plot from the formula.
        """
        self.formula.plots.remove(plot_widget.plot)
        self.plots_container.remove_widget(plot_widget.__self__)


class PropertyDisplayBinding(EventDispatcher):
    """Tracks a property (input/output variable) and updates the widget
    representing the property with the new value.
    """

    prop_from_display_setter = ObjectProperty(lambda x: x)
    """Lambda that can be used to convert the property from the displayed value
    in the GUI when that changes (that could e.g. be a string) to the correct
    type when setting the property value of the formula.
    """

    display_from_prop_setter = ObjectProperty(lambda x: x)
    """Like :attr:`prop_from_display_setter`, but converts the property so it
    can be displayed in the GUI (e.g. to string from float).
    """

    obj = ObjectProperty(None)
    """The object to track.
    """

    obj_prop = StringProperty('')
    """The property of :attr:`obj` to track.
    """

    read_only = BooleanProperty(False)
    """Whether it's read only and cannot be updated from the GUI.
    """

    prop_value = ObjectProperty('')
    """Current value of the property as it's shown in the GUI.
    """

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
        """Callback to set the from the GUI.
        """
        if not self.obj or not self.obj_prop or self.read_only:
            return

        setattr(self.obj, self.obj_prop, self.prop_from_display_setter(value))


class FormulaVariableBehavior(EventDispatcher):
    """Visualization for a formula variable with an attached name.
    """

    name = StringProperty('')
    """The name of the property that is displayed to user.
    """


class PlotWidget(BoxLayout):
    """Widget that visualizes a plot for a formula.
    """

    plot: FormulaPlot = ObjectProperty(None, rebind=True)
    """The plot that visualizes the formula.
    """

    formula_widget: FormulaWidget = ObjectProperty(None)
    """The formula to visualize.
    """

    mouse_x_val = NumericProperty(0)
    """x-pos of the mouse in formula input domain.
    """

    mouse_x2_val = NumericProperty(None, allownone=True)
    """x2-pos of the mouse in formula input domain.
    """

    mouse_y_val = NumericProperty(0)
    """y value at the current mouse pos in formula output domain.
    """

    graph_min_height = NumericProperty(200)
    """Smallest height for the graph.
    """

    _names_to_x_variables = DictProperty({})

    def select_x_variable(self, x_prop, variable_name):
        """Sets the input variable to display on the horizontal/vertical.
        """
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
        """Updates the list of input variables the user can select from in the
        GUI, when selecting the variable to show on an axis.
        """
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
    """The app that shows all the formula.
    """

    theme = ObjectProperty(None, rebind=True)
    """The flat material design style theme to use.
    """

    formula_container_widget = ObjectProperty(None)
    """Widget that contains all the formula.
    """

    focal_len_from_io_f: LensFocalLengthFormula = ObjectProperty(None)
    """Formula that computes focal length and mag from the other
    parameters.
    """

    image_from_f: LensFixedObjectFormula = ObjectProperty(None)
    """Formula that computes the parameters for a fixed object.
    """

    objective_lens: LensFixedObjectFormula = ObjectProperty(None)
    """Formula that computes the parameters for a fixed object.

    This is the first lens in the 4-lens chain.
    """

    cam_lens_further: LensFixedObjectFormula = ObjectProperty(None)
    """Formula that computes the parameters for a fixed object.

    This is the second lens in the 4-lens chain.
    """

    cam_lens_closer: LensFixedObjectFormula = ObjectProperty(None)
    """Formula that computes the parameters for a fixed object.

    This is the third lens in the 4-lens chain.
    """

    cam_lens_closer2: LensFixedObjectFormula = ObjectProperty(None)
    """Formula that computes the parameters for a fixed object.

    This is the forth and final lens in the 4-lens chain.
    """

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
