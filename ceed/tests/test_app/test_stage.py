from .examples.stages import create_test_stages
from ceed.tests.ceed_app import CeedTestApp


async def test_stage_remove_shape_from_all(stage_app: CeedTestApp):
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
