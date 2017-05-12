import numpy as np
import os

from kivy.properties import OptionProperty, ListProperty, ObjectProperty, \
    StringProperty, NumericProperty, DictProperty, BooleanProperty
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.graphics import Color

from ceed.function import CeedFunc, FuncDoneException
from ceed.utils import fix_name
from ceed.shape import get_painter
from ceed.shape import CeedShapeGroup


class StageDoneException(Exception):
    pass


class StageFactoryBase(EventDispatcher):

    stages = []

    show_widgets = False

    stage_names = DictProperty([])

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(StageFactoryBase, self).__init__(**kwargs)
        self.stages = []

    def on_changed(self, *largs):
        pass

    def save_stages(self, id_map=None):
        if id_map is None:
            id_map = {}
        for stage in self.stages:
            for s in stage.get_stages():
                for f in s.functions:
                    CeedFunc.fill_id_map(f, id_map)

        states = [s._copy_state() for s in self.stages]
        return states, id_map

    def recover_stages(self, stages, id_to_func_map=None, old_id_map=None):
        for state in stages:
            stage = CeedStage()
            if self.show_widgets:
                stage.display
            stage._apply_state(state, clone=True, old_id_map=old_id_map)
            self.add_stage(stage)

        id_to_func_map = {} if id_to_func_map is None else id_to_func_map
        funcs = []

        for stage in self.stages:
            for s in stage.get_stages():
                for f in s.functions:
                    CeedFunc.fill_id_to_func_map(f, id_to_func_map)
                    funcs.append(f)

        return funcs, id_to_func_map

    def add_stage(self, stage):
        stage.name = fix_name(stage.name, [s.name for s in self.stages])
        self.stages.append(stage)
        self.stage_names[stage.name] = stage
        if self.show_widgets:
            stage.display.show_stage()
        self.dispatch('on_changed')

    def remove_stage(self, stage):
        self.stages.remove(stage)
        del self.stage_names[stage.name]

        if stage._display:
            stage._display.hide_stage()
        self.dispatch('on_changed')

    def clear_stages(self):
        for stage in self.stages[:]:
            self.remove_stage(stage)

    def remove_shape_from_all(self, obj, shape):
        for stage in self.stages:
            for sub_stage in stage.get_stages():
                for stage_shape in sub_stage.shapes:
                    if stage_shape.shape is shape:
                        sub_stage.remove_shape(stage_shape)
                        break

    def _change_stage_name(self, stage, name):
        if stage.name == name:
            return name

        del self.stage_names[stage.name]
        name = fix_name(name, self.stage_names)
        stage.name = name
        self.stage_names[stage.name] = stage
        return name

    def tick_stage(self, stage_name):
        stage = self.stage_names[stage_name]
        shapes = {s.name: [] for s in get_painter().shapes}
        tick_stage = stage.tick_stage(shapes)
        tick_stage.next()
        while True:
            t = yield
            tick_stage.send(t)

            shape_values = []
            for name, colors in shapes.items():
                shape_values.append((name, colors[:]))
                del colors[:]

            yield shape_values

    def get_shapes_arr_views(self, arr):
        shape_views = {}
        for shape in get_painter().shapes:
            x1, y1, x2, y2 = map(int, shape.bounding_box)
            view = arr[x1:x2, y1:y2]

            index = np.zeros((x2 - x1, y2 - y1), dtype=np.bool)
            for x, y in shape.inside_points:
                index[x, y] = True

            shape_views[shape.name] = view, index
        return shape_views

    def fill_shape_arr_values(self, shape_views, shape_values):
        for name, colors in shape_values:
            r, g, b, a = 0., 0., 0., 1.
            for r2, g2, b2, a2 in colors:
                if r2 is not None:
                    r = r2
                if g2 is not None:
                    g = g2
                if b2 is not None:
                    b = b2
            r = min(max(int(r * 255), 0), 255)
            g = min(max(int(g * 255), 0), 255)
            b = min(max(int(b * 255), 0), 255)

            view, index = shape_views[name]
            view[index] = r, g, b

    def get_shapes_gl_color_instructions(self, canvas, name):
        shape_views = {}
        for shape in get_painter().shapes:
            instructions = shape.add_shape_instructions(
                (0, 0, 0, 1), name, canvas)
            if not instructions:
                raise ValueError('Malformed shape {}'.format(shape.name))

            shape_views[shape.name] = instructions[0]
        return shape_views

    def fill_shape_gl_color_values(self, shape_views, shape_values,
                                   projection=None):
        result = []

        for name, colors in shape_values:
            r = g = b = a = None
            for r2, g2, b2, a2 in colors:
                if r2 is not None:
                    r = r2
                if g2 is not None:
                    g = g2
                if b2 is not None:
                    b = b2
                if a2 is not None:
                    a = a2

            if r is not None:
                r = min(max(r, 0.), 1.)
            if g is not None:
                g = min(max(g, 0.), 1.)
            if b is not None:
                b = min(max(b, 0.), 1.)
            if a is not None:
                a = min(max(a, 0.), 1.)

            color = shape_views[name] if shape_views is not None else None
            if r is None and b is None and g is None and a is None:
                if color is not None:
                    color.rgba = 0, 0, 0, 0
                result.append((name, 0, 0, 0, 0))
            elif projection:
                if a is None:
                    a = 1

                vals = [v for v in (r, g, b) if v is not None]
                if not vals:
                    val = 0
                else:
                    val = sum(vals) / float(len(vals))

                if color is not None:
                    color.a = a
                    setattr(color, projection, val)
                result.append((name, val, val, val, a))
            else:
                r, g, b = r or 0, g or 0, b or 0
                if a is None:
                    a = 1
                if color is not None:
                    color.rgba = r, g, b, a
                result.append((name, r, g, b, a))
        return result

    def remove_shapes_gl_color_instructions(self, canvas, name):
        if canvas:
            canvas.remove_group(name)


class CeedStage(EventDispatcher):

    name = StringProperty('Stage')

    order = OptionProperty('serial', options=['serial', 'parallel'])

    complete_on = OptionProperty('all', options=['all', 'any'])

    stages = ListProperty([])

    parent_stage = ObjectProperty(None, rebind=True, allownone=True)

    functions = ListProperty([])

    shapes = ListProperty([])

    color_r = BooleanProperty(False)

    color_g = BooleanProperty(False)

    color_b = BooleanProperty(True)

    color_a = NumericProperty(None, allownone=True)

    _display = None

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(CeedStage, self).__init__(**kwargs)
        for name in self._copy_state():
            self.fbind(name, self.dispatch, 'on_changed')
        self.fbind('on_changed', StageFactory.dispatch, 'on_changed')

    def on_changed(self, *largs):
        pass

    def _copy_state(self, state={}, recurse=True):
        d = {}
        for name in ('order', 'name', 'color_r', 'color_g', 'color_b',
                     'complete_on'):
            d[name] = getattr(self, name)

        d['stages'] = [s._copy_state() for s in self.stages]
        d['functions'] = [f._copy_state() for f in self.functions]
        d['shapes'] = [s.name for s in self.shapes]
        d.update(state)
        return d

    def _apply_state(self, state={}, clone=False, old_id_map=None):
        stages = state.pop('stages', [])
        functions = state.pop('functions', [])
        shapes_state = state.pop('shapes', [])

        for k, v in state.items():
            setattr(self, k, v)

        for data in stages:
            s = CeedStage()
            if self._display:
                s.display
            s._apply_state(data, clone=True)
            self.add_stage(s)

        for data in functions:
            self.add_func(
                CeedFunc.make_func(data, clone=True, old_id_map=old_id_map))

        shapes = get_painter().shape_names
        groups = get_painter().shape_group_names
        for name in shapes_state:
            if name in shapes:
                self.add_shape(shapes[name])
            elif name in groups:
                self.add_shape(groups[name])

    @property
    def display(self):
        if self._display:
            return self._display

        w = self._display = Factory.StageWidget(stage=self)
        return w

    def add_stage(self, stage):
        self.stages.append(stage)
        stage.parent_stage = self

        if self._display:
            stage.display.show_stage()

    def remove_stage(self, stage):
        self.stages.remove(stage)
        stage.parent_stage = None

        if self._display and stage._display:
            stage._display.hide_stage()

    def add_func(self, func, after=None):
        i = None
        if after is None:
            self.functions.append(func)
        else:
            i = self.functions.index(after)
            self.functions.insert(i + 1, func)
        if self._display:
            self._display.set_func_controller(func.display)
            func.display.show_func(i)

    def remove_func(self, func):
        self.functions.remove(func)
        if func._display:
            func._display.hide_func()

    def add_shape(self, shape):
        if any([s for s in self.shapes if shape.name == s.name]):
            return
        stage_shape = StageShape(stage=self, shape=shape)
        self.shapes.append(stage_shape)
        if self._display:
            stage_shape.display.show_widget()

    def remove_shape(self, stage_shape):
        self.shapes.remove(stage_shape)

        if self._display and stage_shape._display:
            stage_shape.display.hide_widget()

    def get_stages(self):
        yield self
        for stage in self.stages:
            for s in stage.get_stages():
                yield s

    def tick_stage(self, shapes):
        names = set()
        for shape in self.shapes:
            shape = shape.shape
            if isinstance(shape, CeedShapeGroup):
                for shape in shape.shapes:
                    names.add(shape.name)
            else:
                names.add(shape.name)
        names = list(names)

        stages = [s.tick_stage(shapes) for s in self.stages]
        funcs = self.tick_funcs()
        serial = self.order == 'serial'
        end_on_first = self.complete_on == 'any' and not serial
        r, g, b = self.color_r, self.color_g, self.color_b
        a = self.color_a
        for tick_stage in stages[:]:
            tick_stage.next()

        while funcs is not None or stages:
            t = yield
            if funcs is not None:
                try:
                    funcs.next()
                    val = funcs.send(t)
                    values = (val if r else None, val if g else None,
                              val if b else None, a)
                    for name in names:
                        shapes[name].append(values)
                except FuncDoneException:
                    funcs = None
                    if end_on_first:
                        del stages[:]

            for tick_stage in stages[:]:
                try:
                    tick_stage.send(t)
                    if serial:
                        break
                except StageDoneException:
                    if end_on_first:
                        del stages[:]
                        funcs = None
                        break
                    stages.remove(tick_stage)

        raise StageDoneException

    def tick_funcs(self):
        raised = False
        for func in self.functions:
            try:
                if not raised:
                    t = yield
                raised = False
                func.init_func(t)
                yield func(t)
                while True:
                    t = yield
                    yield func(t)
            except FuncDoneException:
                raised = True
        raise FuncDoneException


class StageShape(EventDispatcher):

    shape = ObjectProperty(None, rebind=True)

    stage = ObjectProperty(None, rebind=True)

    name = StringProperty('')

    _display = None

    def __init__(self, **kwargs):
        super(StageShape, self).__init__(**kwargs)
        self.shape.fbind('name', self._update_name)
        self._update_name()

    def _update_name(self, *largs):
        self.name = self.shape.name

    @property
    def display(self):
        if self._display:
            return self._display

        w = self._display = Factory.StageShapeDisplay(stage_shape=self)
        return w

StageFactory = StageFactoryBase()

def _bind_remove(*largs):
    get_painter().fbind('on_remove_shape', StageFactory.remove_shape_from_all)
    get_painter().fbind('on_remove_group', StageFactory.remove_shape_from_all)
if not os.environ.get('KIVY_DOC_INCLUDE', None):
    Clock.schedule_once(_bind_remove, 0)
