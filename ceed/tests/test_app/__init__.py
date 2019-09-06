
__all__ = ('replace_text', 'touch_widget')


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


async def touch_widget(app, widget):
    async for _ in app.do_touch_down_up(widget=widget):
        pass
    await app.wait_clock_frames(2)
