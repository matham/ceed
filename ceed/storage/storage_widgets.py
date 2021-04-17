"""Storage widgets
==================

Widgets visualizing data storage.

"""

from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.properties import NumericProperty, BooleanProperty, ObjectProperty, \
    StringProperty, ListProperty

from more_kivy_app.utils import yaml_dumps

from ceed.graphics import ShowMoreBehavior
from ceed.storage.controller import CeedDataWriterBase

__all__ = (
    'LogWidgetBase', 'StageLogWidget', 'ImageLogWidget', 'ExperimentLogWidget')


class ExperimentLogWidget(BoxLayout):
    """Displays data associated with all the experiments.
    """

    data_storage: CeedDataWriterBase = None
    """The :class:`~ceed.storage.controller.CeedDataWriterBase` used by Ceed.
    """

    num_experiments = NumericProperty(0)

    num_images = NumericProperty(0)
    """The number of images recorded to the file.
    """

    log_text = StringProperty('')
    """The text logged to the file.
    """

    log_container = None
    """Widget that displays the logs.
    """

    config_str = StringProperty('')
    """The yaml encoded string containing the current app Ceed config data.
    """

    mea_config = StringProperty('')
    """String representation of the mea configuration matrix
    :attr:`~ceed.view.controller.ViewControllerBase.mea_transform`.
    """

    experiment_names = ListProperty([])
    """List of all the names of the experiments run so far. The names are how
    they are stored in the file.
    """

    selected_config_str = StringProperty('')
    """The yaml encoded string containing the Ceed config data of the experiment
    or app selected in the GUI.
    """

    selected_mea_config_str = StringProperty('')
    """String representation of the mea configuration matrix
    :attr:`~ceed.view.controller.ViewControllerBase.mea_transform` of the
    experiment or app selected in the GUI.
    """

    _bound_config = None, None, None

    def __init__(self, **kwargs):
        super(ExperimentLogWidget, self).__init__(**kwargs)

        data_storage = self.data_storage = App.get_running_app().ceed_data
        data_storage.fbind(
            'on_experiment_change', self.experiment_change_callback)

        def set_config(*largs):
            self.selected_config_str = self.config_str
        config_uid = self.fbind('config_str', set_config)
        set_config()

        def set_mea_config(*largs):
            self.selected_mea_config_str = self.mea_config
        mea_config_uid = self.fbind('mea_config', set_mea_config)
        set_mea_config()
        self._bound_config = self, config_uid, mea_config_uid

    def bind_to_current_config_selection(self, name):
        """Binds and tracks the config of the given experiment or app and shows
        it in the GUI.
        """
        obj, uid, mea_uid = self._bound_config
        obj.unbind_uid('config_str', uid)
        obj.unbind_uid('mea_config', mea_uid)

        if name == 'app':
            obj = self
        else:
            number = name.split(' ')[1]
            for widget in self.log_container.children:
                if isinstance(widget, StageLogWidget):
                    if widget.experiment_number == number:
                        obj = widget
                        break
            else:
                assert False

        def set_config(*largs):
            self.selected_config_str = obj.config_str
        config_uid = obj.fbind('config_str', set_config)
        set_config()

        def set_mea_config(*largs):
            self.selected_mea_config_str = obj.mea_config
        mea_config_uid = obj.fbind('mea_config', set_mea_config)
        set_mea_config()
        self._bound_config = obj, config_uid, mea_config_uid

    def experiment_change_callback(self, instance, name, value):
        """Called in response to any data related changes (experiment
        started/finished, image saved, etc.) and updates the GUI to show the
        new data.
        """
        if name == 'open':
            self.handle_open_file()
        elif name == 'close':
            self.log_text = ''
            self.experiment_names = []
            for widget in self.log_container.children[:]:
                self.log_container.remove_widget(widget)
        elif name == 'app_log':
            self.log_text = self.data_storage.get_log_data()
        elif name == 'image_add':
            data = self.data_storage.get_saved_image(value)
            widget = ImageLogWidget(data_storage=self.data_storage, **data)
            self.log_container.add_widget(widget)
        elif name == 'experiment_ended':
            for widget in self.log_container.children:
                if isinstance(widget, StageLogWidget):
                    if widget.experiment_number == value:
                        widget.refresh_metadata()
                        self.mark_experiments_with_changed_config()
                        break
            else:
                assert False
        elif name == 'experiment_stop':
            data = self.data_storage.get_experiment_metadata(value)
            widget = StageLogWidget(data_storage=self.data_storage, **data)
            self.log_container.add_widget(widget)
            self.mark_experiments_with_changed_config()
            self.experiment_names.append('Exp {}'.format(value))
        elif name == 'experiment_notes':
            for widget in self.log_container.children:
                if isinstance(widget, StageLogWidget):
                    if widget.experiment_number == value:
                        widget.notes = self.data_storage.get_experiment_notes(
                            value)
                        break
            else:
                assert False
        elif name == 'app_config':
            self.format_config(self.data_storage.read_config())
            self.mark_experiments_with_changed_config()
        elif name == 'experiment_mea_settings':
            for widget in self.log_container.children:
                if isinstance(widget, StageLogWidget):
                    if widget.experiment_number == value:
                        widget.config = self.data_storage.get_experiment_config(
                            value)
                        widget.mea_config = \
                            self.data_storage.get_config_mea_matrix_string(
                                value)
                        self.mark_experiments_with_changed_config()
                        break
            else:
                assert False
        else:
            assert False

    def handle_open_file(self):
        """Updates GUI when a new data file is opened.
        """
        data_storage = self.data_storage
        self.log_text = data_storage.get_log_data()

        experiment_numbers = data_storage.get_experiment_numbers()
        num_images = data_storage.get_num_fluorescent_images()

        items = []
        for num in experiment_numbers:
            items.append(data_storage.get_experiment_metadata(num))
        for i in range(num_images):
            items.append(data_storage.get_saved_image(i))

        items = sorted(items, key=lambda item: item['save_time'])
        for item in items:
            cls = StageLogWidget if 'stage' in item else ImageLogWidget
            widget = cls(data_storage=data_storage, **item)
            self.log_container.add_widget(widget)

        self.experiment_names = [
            'Exp {}'.format(num) for num in experiment_numbers]

        self.mark_experiments_with_changed_config()

    def mark_experiments_with_changed_config(self):
        """Visually points out in the data log any experiment whose mea-related
        config is different than the previous experiment. This helps the user
        know if they need to backport mea config changes that happened during
        an experiment to the experiment after it is done.
        """
        widgets = [
            w for w in reversed(self.log_container.children)
            if isinstance(w, StageLogWidget)]
        if not widgets:
            return

        for i, widget in enumerate(widgets[:-1]):
            widget.mea_config_different = \
                widget.mea_config != widgets[i + 1].mea_config

        widgets[-1].mea_config_different = \
            widgets[-1].mea_config != self.mea_config

    def format_config(self, config):
        """Converts the config into a yaml-encoded string and the mea config
        to a string showing the matrix.
        """
        s = ''
        for key, value in config.items():
            key = str(key)
            s += '[b]' + key + '[/b]\n'
            s += '-' * len(key) + '\n'
            s += yaml_dumps(value) + '\n\n'
        self.config_str = s

        self.mea_config = self.data_storage.get_config_mea_matrix_string()

    def copy_mea_config_to_exp(self, source, target):
        """Copies the mea config values from one experiment to another.
        """
        if source == target:
            return

        source_block = None
        if source != 'app':
            source_block = source.split(' ')[1]

        target_block = target.split(' ')[1]

        s_config = self.data_storage.get_config_mea_matrix_string(source_block)
        t_config = self.data_storage.get_config_mea_matrix_string(target_block)
        if s_config == t_config:
            return

        self.data_storage.set_config_mea_matrix_string(
            target_block, t_config, s_config)
        self.mark_experiments_with_changed_config()


class LogWidgetBase:
    """Base class for log items displayed in the GUI.
    """

    data_storage = None
    """The :class:`~ceed.storage.controller.CeedDataWriterBase` used by Ceed.
    """

    image_widget = None
    """Widget that shows an image.
    """

    save_time = NumericProperty(0)
    """The time the item was saved.
    """

    notes = StringProperty('')
    """Any notes associated with the item.
    """

    image = ObjectProperty(None)
    """The image object recorded in the log item, if any.
    """

    def __init__(self, data_storage=None, **kwargs):
        super(LogWidgetBase, self).__init__(**kwargs)
        self.data_storage = data_storage
        self.fbind('image', self.show_image)
        self.show_image()

    def show_image(self, *largs):
        """Displays the :attr:`image` in the log item.
        """
        self.image_widget.update_img(self.image)


class StageLogWidget(LogWidgetBase, ShowMoreBehavior, BoxLayout):
    """Log item representing a recorded experiment."""

    experiment_number = StringProperty('')
    """The experiment number as saved in the file.
    """

    stage = StringProperty('')
    """The name of the stage used in the experiment.
    """

    duration_frames = NumericProperty(0)
    """The number of frames in the experiment.
    """

    duration_sec = NumericProperty(0)
    """The duration of the experiment.
    """

    config = ObjectProperty({})
    """The app config used for the experiment.
    """

    config_str = StringProperty('')
    """String representation of the config.
    """

    mea_config = StringProperty('')
    """String representation of
    :attr:`~ceed.view.controller.ViewControllerBase.mea_transform` used in the
    experiment.
    """

    mea_config_different = BooleanProperty(False)
    """Whether the
    :attr:`~ceed.view.controller.ViewControllerBase.mea_transform` used in this
    experiment is different from the next experiment, or app if it's the last.
    """

    def refresh_metadata(self):
        """Updates the instance properties from the saved data.
        """
        data = self.data_storage.get_experiment_metadata(self.experiment_number)
        for key, value in data.items():
            setattr(self, key, value)

    def update_text(self, text):
        """Updates the notes from the GUI.
        """
        if text == self.notes:
            return
        self.data_storage.set_experiment_notes(self.experiment_number, text)

    def on_config(self, *largs):
        s = ''
        for key, value in self.config.items():
            key = str(key)
            s += '[b]' + key + '[/b]\n'
            s += '-' * len(key) + '\n'
            s += yaml_dumps(value) + '\n\n'
        self.config_str = s


class ImageLogWidget(LogWidgetBase, ShowMoreBehavior, BoxLayout):
    """Log item representing a recorded image from the player."""

    image_num = NumericProperty(0)
    """The number of the image as saved in the file.
    """
