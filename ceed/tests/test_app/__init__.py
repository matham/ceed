import os

__all__ = (
    'replace_text', 'touch_widget', 'select_spinner_value', 'escape',
    'run_plugin_experiment')


async def replace_text(app, text_widget, new_text):
    # activate it
    async for _ in app.do_touch_down_up(widget=text_widget):
        pass

    # select all
    ctrl_it = app.do_keyboard_key(key='lctrl', modifiers=['ctrl'])
    await ctrl_it.__anext__()  # down
    async for _ in app.do_keyboard_key(key='a', modifiers=['ctrl']):
        pass
    await ctrl_it.__anext__()  # up

    # replace text
    for key in ['delete'] + list(new_text) + ['enter']:
        async for _ in app.do_keyboard_key(key=key):
            pass

    await app.wait_clock_frames(2)


async def touch_widget(app, widget):
    async for _ in app.do_touch_down_up(widget=widget):
        pass
    await app.wait_clock_frames(2)


async def select_spinner_value(func_app, func_name, spinner):
    from kivy.metrics import dp
    async for _ in func_app.do_touch_down_up(
            pos=spinner.to_window(spinner.x + dp(30), spinner.center_y)):
        pass
    await func_app.wait_clock_frames(2)

    label = func_app.resolve_widget().down(text=func_name)()
    await touch_widget(func_app, label)
    await func_app.wait_clock_frames(2)


async def escape(app):
    async for _ in app.do_keyboard_key(key='escape'):
        pass
    await app.wait_clock_frames(2)


async def run_plugin_experiment(
        ceed_app, tmp_path, external, func=None, stage=None):
    from .examples.shapes import CircleShapeP1
    from .examples.stages import SerialAllStage
    from .examples.experiment import wait_experiment_done, \
        wait_experiment_stopped

    stage_factory = ceed_app.app.stage_factory

    shape = CircleShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=True)
    if func is None:
        func = ceed_app.app.function_factory.get('LinearFunc')(
                function_factory=ceed_app.app.function_factory, duration=1)
    if stage is None:
        stage = SerialAllStage(
            stage_factory=stage_factory, show_in_gui=True, app=ceed_app,
            create_add_to_parent=False,
            stage_cls=stage_factory.get('CeedStage'))

    stage.stage.add_shape(shape.shape)
    stage.stage.add_func(func)
    await ceed_app.wait_clock_frames(2)

    if external:
        p = os.environ.get('PYTHONPATH', '')
        os.environ['PYTHONPATH'] = p + os.pathsep + str(tmp_path)
        ceed_app.app.view_controller.start_process()
        if p:
            os.environ['PYTHONPATH'] = p
        else:
            del os.environ['PYTHONPATH']

    await ceed_app.wait_clock_frames(2)

    ceed_app.app.view_controller.request_stage_start(stage.name)
    await wait_experiment_done(ceed_app)

    if external:
        ceed_app.app.view_controller.stop_process()
    await ceed_app.wait_clock_frames(2)
    await wait_experiment_stopped(ceed_app)
