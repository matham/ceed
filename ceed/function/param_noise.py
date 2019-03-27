import random

from kivy.event import EventDispatcher
from kivy.properties import NumericProperty


class ParameterNoiseFactory(EventDispatcher):

    noise_classes = {}

    def __init__(self, **kwargs):
        super(ParameterNoiseFactory, self).__init__(**kwargs)
        self.noise_classes = {}

    def register_class(self, cls):
        self.noise_classes[cls.__name__] = cls

    def get_cls(self, name):
        return self.noise_classes[name]

    def make_instance(self, config):
        cls = self.get_cls(config['cls'])
        instance = cls(**{k: v for k, v in config.items() if k != 'cls'})
        return instance


class NoiseBase(object):

    def sample(self):
        raise NotImplementedError

    def get_config(self):
        return {'cls': self.__class__.__name__}


class GaussianNoise(NoiseBase):

    min_val = NumericProperty(0)

    max_val = NumericProperty(1)

    mean_val = NumericProperty(0.5)

    stdev = NumericProperty(.1)

    def sample(self):
        val = random.gauss(self.mean_val, self.stdev)
        return max(min(val, self.max_val), self.min_val)

    def get_config(self):
        config = super(GaussianNoise, self).get_config()
        for attr in ('min_val', 'max_val', 'mean_val', 'stdev'):
            config[attr] = getattr(self, attr)
        return config


class UniformNoise(NoiseBase):

    min_val = NumericProperty(0)

    max_val = NumericProperty(1)

    def sample(self):
        return random.uniform(self.min_val, self.max_val)

    def get_config(self):
        config = super(UniformNoise, self).get_config()
        for attr in ('min_val', 'max_val'):
            config[attr] = getattr(self, attr)
        return config


def register_noise_classes(noise_factory):
    noise_factory.register_class(GaussianNoise)
    noise_factory.register_class(UniformNoise)
