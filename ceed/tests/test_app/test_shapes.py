from ceed.tests.conftest import CeedTestApp


async def test_simple_shape(ceed_app: CeedTestApp):
    from kivy.uix.behaviors.knspace import knspace
    painter = knspace.painter
    assert not ceed_app.shape_factory.shapes
    shape = painter.create_add_shape(
        'polygon', points=[0, 0, 300, 0, 300, 800, 0, 800], name='a shape')
    await ceed_app.wait_clock_frames(2)

    assert shape in ceed_app.shape_factory.shapes
    assert shape.name == 'a shape'
    v1, v2, v3 = ceed_app.get_widget_pos_pixel(
        painter, [(0, 0), (299, 799), (200, 200)])
    assert max(v1[:3]) > 50
    assert max(v2[:3]) > 50
    assert max(v3[:3]) == 0

    painter.remove_shape(shape)
    await ceed_app.wait_clock_frames(2)

    assert shape not in ceed_app.shape_factory.shapes
    assert not ceed_app.shape_factory.shapes
    v1, v2, v3 = ceed_app.get_widget_pos_pixel(
        painter, [(0, 0), (299, 799), (200, 200)])
    assert max(v1[:3]) == 0
    assert max(v2[:3]) == 0
    assert max(v3[:3]) == 0

    painter.add_shape(shape)
    assert shape in ceed_app.shape_factory.shapes
    assert shape.name == 'a shape'

    assert not shape.locked
    painter.lock_shape(shape)
    assert shape.locked
    painter.unlock_shape(shape)
    assert not shape.locked


async def test_multiple_shapes(ceed_app: CeedTestApp):
    from kivy.uix.behaviors.knspace import knspace
    painter = knspace.painter
    assert not ceed_app.shape_factory.shapes

    painter.create_add_shape(
        'polygon', points=[0, 0, 300, 0, 300, 800, 0, 800], name='some shape')
    painter.create_add_shape(
        'polygon', points=[0, 900, 300, 900, 300, 1200, 0, 1200],
        name='other shape')
    await ceed_app.wait_clock_frames(5)
