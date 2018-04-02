import numpy as np
import nixio as nix
from cplcom.utils import yaml_dumps, yaml_loads
from cplcom.config import apply_config
from ffpyplayer.pic import Image

from ceed.function import FunctionFactoryBase, register_all_functions
from ceed.stage import StageFactoryBase
from ceed.shape import CeedPaintCanvasBehavior
from ceed.storage.controller import DataSerializerBase
from ceed.view.controller import ViewControllerBase


class CeedDataReader(object):

    filename = ''

    shapes_intensity = {}

    electrodes_data = {}

    electrodes_metadata = {}

    electrode_dig_data = None

    electrode_intensity_alignment = None

    led_state = None

    view_controller = None

    data_serializer = None

    function_factory = None

    stage_factory = None

    shape_factory = None

    experiment_stage_name = ''

    ceed_version = ''

    experiment = 0

    _nix_file = None

    _block = None

    def __init__(self, filename, **kwargs):
        super(CeedDataReader, self).__init__(**kwargs)
        self.filename = filename

    def open_h5(self):
        if self._h5_file is not None:
            raise Exception('File already open')

        self._nix_file = nix.File.open(
            self.filename, nix.FileMode.ReadOnly)

    def read_experiment(self, experiment):
        self._block = block = self._nix_file.blocks[
            'experiment{}'.format(experiment)]
        section = self._nix_file.sections[
            'experiment{}_metadata'.format(experiment)]
        self.experiment = experiment

        self.experiment_stage_name = section['stage']
        self.ceed_version = section['ceed_version']
        config = section.sections['app_config']

        config_data = {}
        for prop in config.props:
            config_data[prop.name] = yaml_loads(prop.values[0].value)

        view = self.view_controller = ViewControllerBase()
        ser = self.data_serializer = DataSerializerBase()
        func = self.function_factory = FunctionFactoryBase()
        register_all_functions(func)
        shape = self.shape_factory = CeedPaintCanvasBehavior(knsname='painter')
        stage = self.stage_factory = StageFactoryBase(
            function_factory=func, shape_factory=shape)

        apply_config(config_data, {
            'view': view, 'serializer': ser, 'function': func})
        self.populate_config(config_data, shape, func, stage)

        data = self.shapes_intensity = {}
        for item in block.data_arrays:
            if not item.name.startswith('shape_'):
                continue
            data[item.name[6:]] = item

        self.led_state = block.data_arrays['led_state']

        self.electrode_intensity_alignment = self._nix_file.blocks[
            'ceed_mcs_alignment'].data_arrays[
            'experiment_{}'.format(experiment)]

        mcs_block = self._nix_file.blocks['mcs_data']
        mcs_metadata = mcs_block.metadata
        self.electrode_dig_data = None
        if 'digital_io' in mcs_block.data_arrays:
            self.electrode_dig_data = mcs_block.data_arrays['digital_io']

        electrode_data = self.electrodes_data = {}
        electrodes_metadata = self.electrodes_metadata = {}
        for item in mcs_block.data_arrays:
            if item.name.startswith('electrode_'):
                electrode_data[item.name[10:]] = item

            electrodes_metadata[item.name[10:]] = electrode_metadata = {}
            for prop in mcs_metadata.sections[item.name].props:
                electrode_metadata[prop.name] = yaml_loads(prop.values[0].value)

    def get_fluorescent_image(self):
        return self.read_fluorescent_image_from_block(self._block)

    def get_electrode_offset_scale(self, electrode):
        metadata = self.electrodes_metadata[electrode]
        return (
            float(metadata['ADZero']),
            float(metadata['ConversionFactor']) *
            10. ** float(metadata['Exponent']))

    @staticmethod
    def populate_config(
            settings, shape_factory, function_factory, stage_factory):
        old_to_new_name_shape_map = {}
        shape_factory.set_state(settings['shape'], old_to_new_name_shape_map)

        id_to_func_map = {}
        old_id_map = {}
        old_to_new_name = {}

        f1 = function_factory.recover_funcs(
            settings['function'], id_to_func_map, old_id_map, old_to_new_name)
        f2 = stage_factory.recover_stages(
            settings['stage'], id_to_func_map, old_id_map=old_id_map,
            old_to_new_name_shape_map=old_to_new_name_shape_map)

        old_to_new_id_map = {v: k for k, v in old_id_map.items()}

        id_map = settings['func_id_map']
        id_map_new = {old_to_new_id_map[a]: old_to_new_id_map[b]
                      for a, b in id_map.items()
                      if a in old_to_new_id_map and b in old_to_new_id_map}

        for f in f1[0] + f2[0]:
            for func in f.get_funcs():
                func.finalize_func_state(
                    id_map_new, id_to_func_map, old_to_new_name)

    @staticmethod
    def read_fluorescent_image_from_block(block):
        try:
            group = block.groups['fluorescent_image']
        except KeyError:
            return None

        planes = [np.array(d).tobytes() for d in group.data_arrays]
        img = Image(plane_buffers=planes, pix_fmt=group.metadata['pix_fmt'],
                    size=yaml_loads(group.metadata['size']))
        return img

