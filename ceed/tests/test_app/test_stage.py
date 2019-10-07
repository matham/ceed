from .examples.stages import create_test_stages, make_stage
from ceed.tests.ceed_app import CeedTestApp
from ceed.tests.test_app import replace_text, touch_widget


async def test_stage_find_shape_in_all_stages(stage_app: CeedTestApp):
    (s1, s2, s3), (group, shape1, shape2, shape3) = create_test_stages(
        stage_app=stage_app, manually_add=True)
    await stage_app.wait_clock_frames(2)

    for shape in (shape1, shape2, shape3):
        for stage in (s1, s2, s3):
            assert shape.shape in [s.shape for s in stage.stage.shapes]
        assert shape.shape in group.shapes

    stage_app.shape_factory.remove_shape(shape2.shape)
    await stage_app.wait_clock_frames(2)

    for shape in (shape1, shape3):
        for stage in (s1, s2, s3):
            assert shape.shape in [s.shape for s in stage.stage.shapes]
        assert shape.shape in group.shapes
    for shape in (shape2, ):
        for stage in (s1, s2, s3):
            assert shape.shape not in [s.shape for s in stage.stage.shapes]
        assert shape.shape not in group.shapes

    stage_app.shape_factory.remove_shape(shape1.shape)
    await stage_app.wait_clock_frames(2)

    for shape in (shape3, ):
        for stage in (s1, s2, s3):
            assert shape.shape in [s.shape for s in stage.stage.shapes]
        assert shape.shape in group.shapes
    for shape in (shape2, shape1):
        for stage in (s1, s2, s3):
            assert shape.shape not in [s.shape for s in stage.stage.shapes]
        assert shape.shape not in group.shapes

    stage_app.shape_factory.remove_shape(shape3.shape)
    await stage_app.wait_clock_frames(2)

    for shape in (shape2, shape1, shape3):
        for stage in (s1, s2, s3):
            assert shape.shape not in [s.shape for s in stage.stage.shapes]
        assert shape.shape not in group.shapes


async def test_add_empty_stage(stage_app: CeedTestApp):
    stage_factory = stage_app.stage_factory
    assert not stage_factory.stages
    assert not stage_factory.stage_names

    # add first empty stage
    add_stage = stage_app.resolve_widget().down(test_name='add empty stage')()
    await touch_widget(stage_app, add_stage)

    assert stage_factory.stages
    stage = stage_factory.stages[0]
    assert list(stage_factory.stage_names.values()) == [stage]
    assert stage.display.show_more

    # select the stage and add stage to it
    name_label = stage_app.resolve_widget(stage.display).down(
        test_name='stage label')()
    assert not stage.display.selected

    await touch_widget(stage_app, name_label)
    assert stage.display.selected
    await touch_widget(stage_app, add_stage)
    assert stage_factory.stages == [stage]

    # deselect the stage and add stage globally
    assert stage.display.selected
    await touch_widget(stage_app, name_label)
    await touch_widget(stage_app, add_stage)

    assert len(stage_factory.stages) == 2
    assert stage_factory.stages[0] is stage
