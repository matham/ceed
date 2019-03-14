from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.properties import NumericProperty, BooleanProperty, ObjectProperty, \
    StringProperty

from cplcom.utils import yaml_dumps

from ceed.graphics import ShowMoreBehavior
from ceed.storage.controller import CeedDataWriterBase


class ExperimentLogWidget(BoxLayout):

    data_storage = None  # type: CeedDataWriterBase

    num_experiments = NumericProperty(0)

    num_images = NumericProperty(0)

    log_text = StringProperty('')

    log_container = None

    config_str = StringProperty('')

    def __init__(self, **kwargs):
        super(ExperimentLogWidget, self).__init__(**kwargs)
        data_storage = self.data_storage = App.get_running_app().ceed_data
        data_storage.fbind(
            'on_experiment_change', self.experiment_change_callback)

    def experiment_change_callback(self, instance, name, value):
        if name == 'open':
            self.handle_open_file()
        elif name == 'close':
            self.log_text = ''
            for widget in self.log_container.children[:]:
                self.remove_widget(widget)
        elif name == 'app_log':
            self.log_text = self.data_storage.get_log_data()
        elif name == 'image_add':
            data = self.data_storage.get_saved_image(value)
            widget = ImageLogWidget(data_storage=self.data_storage, **data)
            self.log_container.add_widget(widget)
        elif name == 'experiment_ended':
            for widget in self.log_container.children:
                if isinstance(widget, StageLogWidget):
                    print('widget', widget.experiment_number, value)
                    if widget.experiment_number == value:
                        widget.refresh_metadata()
                        break
            else:
                assert False
        elif name == 'experiment_stop':
            data = self.data_storage.get_experiment_metadata(value)
            widget = StageLogWidget(data_storage=self.data_storage, **data)
            self.log_container.add_widget(widget)
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
        else:
            assert False

    def handle_open_file(self):
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

    def format_config(self, config):
        s = ''
        for key, value in config.items():
            key = str(key)
            s += '[b]' + key + '[/b]\n'
            s += '-' * len(key) + '\n'
            s += yaml_dumps(value) + '\n\n'
        self.config_str = s


class LogWidgetBase(object):

    data_storage = None

    image_widget = None

    save_time = NumericProperty(0)

    notes = StringProperty('')

    image = ObjectProperty(None)

    def __init__(self, data_storage=None, **kwargs):
        super(LogWidgetBase, self).__init__(**kwargs)
        self.data_storage = data_storage
        self.fbind('image', self.show_image)
        self.show_image()

    def show_image(self, *largs):
        self.image_widget.update_img(self.image)


class StageLogWidget(LogWidgetBase, ShowMoreBehavior, BoxLayout):

    experiment_number = StringProperty('')

    stage = StringProperty('')

    duration_frames = NumericProperty(0)

    duration_sec = NumericProperty(0)

    config = ObjectProperty({})

    config_str = StringProperty('')

    def refresh_metadata(self):
        data = self.data_storage.get_experiment_metadata(self.experiment_number)
        print('refresh', data)
        for key, value in data.items():
            setattr(self, key, value)

    def update_text(self, text):
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

    image_num = NumericProperty(0)
