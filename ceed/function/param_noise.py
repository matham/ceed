"""Parameter randomization
==========================

Provides the optional randomization for the parameters of a
:class:`~ceed.function.FuncBase`. Each parameter of the function may be
randomized according to :attr:`~ceed.function.FuncBase.noisy_parameters`.

This module provides a :class:`ParameterNoiseFactory` used to register noise
type classes and some built in noise types.
"""
from typing import Dict, Type, TypeVar

from kivy.event import EventDispatcher
from kivy.properties import NumericProperty, BooleanProperty

__all__ = ('ParameterNoiseFactory', 'NoiseBase', 'NoiseType')


NoiseType = TypeVar('NoiseType', bound='NoiseBase')
"""The type-hint type for :class:`NoiseBase`.
"""


class ParameterNoiseFactory(EventDispatcher):
    """Factory where noise classes are registered and accessed by name.
    """

    noise_classes: Dict[str, Type[NoiseType]] = {}
    """Keeps all classes registered with :meth:`register_class`.
    """

    def __init__(self, **kwargs):
        super(ParameterNoiseFactory, self).__init__(**kwargs)
        self.noise_classes = {}

    def register_class(self, cls: Type[NoiseType]):
        """Registers a :class:`NoiseBase` subclass, with the name of the class
        in :attr:`noise_classes`.
        """
        self.noise_classes[cls.__name__] = cls

    def get_cls(self, name: str) -> Type[NoiseType]:
        """Looks up a noise class by name and returns it.
        """
        return self.noise_classes[name]

    def make_instance(self, config: dict) -> 'NoiseBase':
        """Takes a noise instance's config, as returned by
        :meth:`NoiseBase.get_config`, and creates a noise instance of that
        class and config, and returns it.
        """
        cls = self.get_cls(config['cls'])
        instance = cls(**{k: v for k, v in config.items() if k != 'cls'})
        return instance


class NoiseBase(EventDispatcher):
    """Base class that can be used to randomize a function parameter with
    :attr:`~ceed.function.FuncBase.noisy_parameters`.

    Instances have a :meth:`sample` method that returns a random value when
    called. This is used to sample a new value for function parameters.
    """

    lock_after_forked: bool = BooleanProperty(False)
    """Functions can reference other function. After the reference functions
    are expanded and copied before running the stage as an experiment, all
    randomized parameters whose :attr:`lock_after_forked` is False are
    resampled.

    This allows the parameters with :attr:`lock_after_forked` set to True to
    share the same random value as the original referenced function's
    randomized value.

    See :meth:`ceed.stage.CeedStage.copy_and_resample` for details.
    """

    def sample(self) -> float:
        """Samples the distribution and returns a new value.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """The name of the class.

        This is the name used with :attr:`ParameterNoiseFactory.get_cls`.
        """
        return self.__class__.__name__

    def get_config(self) -> dict:
        """Returns a dict representation of the instance that can be then
        be used to reconstruct it with
        :meth:`ParameterNoiseFactory.make_instance`.

        This is also used to display the instance parameters to the user.
        We infer the type of each parameter from the property value.
        """
        return {
            'cls': self.name,
            'lock_after_forked': self.lock_after_forked}

    def get_prop_pretty_name(self) -> Dict[str, str]:
        """Returns a dict mapping names of the parameters used by the class
        to a nicer representation shown to the user.
        """
        return {'lock_after_forked': 'Lock after fork'}
