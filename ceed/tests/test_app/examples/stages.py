from typing import List

from ceed.tests.ceed_app import CeedTestApp
from ceed.stage import CeedStage, StageFactoryBase
from ceed.function import FuncBase
from ceed.shape import CeedShape
from ceed.tests.test_app.examples.shapes import Shape
from ceed.tests.test_app.examples.funcs import Function


def assert_stages_same(
        stage1: CeedStage, stage2: CeedStage, compare_name=False):
    assert isinstance(stage1, stage2.__class__)

    keys = set(stage1.get_state().keys()) | set(stage2.get_state().keys())
    assert 'name' in keys
    if not compare_name:
        keys.remove('name')

    for key in keys:
        if key in ('stages', 'functions', 'shapes', 'cls'):
            continue
        assert getattr(stage1, key) == getattr(stage2, key)


class StageWrapper(object):

    stage: CeedStage = None

    app: CeedTestApp = None

    stage_factory: StageFactoryBase = None

    stages_container = None

    name = ''

    order = ''

    complete_on = ''

    stages: 'List[StageWrapper]' = []

    parent_wrapper_stage: 'StageWrapper' = None

    functions: List[FuncBase] = []

    shapes: List[Shape] = []

    color_r = False

    color_g = False

    color_b = False

    color_a = None

    def __init__(
            self, app: CeedTestApp = None,
            stage_factory: StageFactoryBase = None,
            parent_wrapper_stage: 'StageWrapper' = None,
            manually_add=True, shapes=[], functions=[]):
        self.stages = []
        super().__init__()
        self.app = app
        if app is None:
            self.stage_factory = stage_factory
        else:
            self.stage_factory = app.stage_factory
            self.stages_container = app.stages_container
        self.parent_wrapper_stage = parent_wrapper_stage
        if parent_wrapper_stage is not None:
            parent_wrapper_stage.stages.append(self)
        self.shapes = shapes
        self.functions = functions

        if manually_add:
            self.manually_add()

    def create_stage(self):
        stage = self.stage = CeedStage(
            stage_factory=self.stage_factory,
            function_factory=self.stage_factory.function_factory,
            shape_factory=self.stage_factory.shape_factory,
            name=self.name, order=self.order, complete_on=self.complete_on,
            color_r=self.color_r, color_g=self.color_g, color_b=self.color_b,
            color_a=self.color_a
        )

        for shape in self.shapes:
            stage.add_shape(shape.shape)

        for func in self.functions:
            stage.add_func(func)

        if self.parent_wrapper_stage is not None:
            self.parent_wrapper_stage.stage.add_stage(stage)

    def manually_add(self):
        self.create_stage()
        if self.parent_wrapper_stage is None:
            self.stages_container.show_stage(self.stage)
        else:
            self.stages_container.show_sub_stage(
                self.stage, self.parent_wrapper_stage.stage)

    def assert_init(self):
        assert self.name == self.stage.name
        assert self.order == self.stage.order
        assert self.complete_on == self.stage.complete_on
        assert self.color_r == self.stage.color_r
        assert self.color_g == self.stage.color_g
        assert self.color_b == self.stage.color_b
        assert self.color_a == self.stage.color_a


class ParaAllStage(StageWrapper):

    name = 'a parallel stage'

    order = 'parallel'

    complete_on = 'all'

    color_r = True

    color_g = True


class ParaAnyStage(StageWrapper):

    name = 'a parallel page'

    order = 'parallel'

    complete_on = 'any'

    color_g = True


class SerialAllStage(StageWrapper):

    name = 'a serial stage'

    order = 'serial'

    complete_on = 'all'

    color_r = True

    color_g = True


class SerialAnyStage(StageWrapper):

    name = 'a serial page'

    order = 'serial'

    complete_on = 'any'

    color_g = True
