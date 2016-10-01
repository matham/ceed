from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty
from kivy.core.window import Window

from ceed.utils import WidgetList, ShowMoreSelection, BoxSelector, \
    ShowMoreBehavior, fix_name


class StageList(ShowMoreSelection, WidgetList, BoxLayout):
    pass
