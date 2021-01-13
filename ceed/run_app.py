# Icon made by https://www.flaticon.com/authors/freepik from
# https://www.flaticon.com/

if __name__ == '__main__':
    import os
    if os.environ.get('COVERAGE_PROCESS_START', None) == '1':
        import coverage
        coverage.process_startup()

    import multiprocessing
    multiprocessing.freeze_support()
    from ceed.main import run_app
    app = run_app()

    from kivy.core.window import Window
    for child in Window.children[:]:
        Window.remove_widget(child)
    from kivy.logger import LoggerHistory
    LoggerHistory.clear_history()
    import gc
    import weakref
    app = weakref.ref(app)
    gc.collect()
    import logging

    if app() is not None and False:
        logging.error('Memory leak: failed to release app for test ')
        import objgraph
        objgraph.show_backrefs(
            [app()], filename=r'E:\backrefs.png', max_depth=100,
            too_many=1)
        # objgraph.show_chain(
        #     objgraph.find_backref_chain(
        #         app(), objgraph.is_proper_module),
        #     filename=r'E:\chain.png')
    # assert app() is None
