
__all__ = ('replace_text', 'touch_widget', 'select_spinner_value', 'escape')


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
