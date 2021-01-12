from ceed.tests.ceed_app import CeedTestApp
import pytest

pytestmark = pytest.mark.ceed_app


@pytest.mark.parametrize(
    "ceed_app", [{'persist_config': 'base1_'}, ], indirect=True)
async def test_change_config_write(ceed_app: CeedTestApp):
    # the default is True
    assert ceed_app.app.view_controller.mirror_mea
    ceed_app.app.view_controller.mirror_mea = False
    assert not ceed_app.app.view_controller.mirror_mea


@pytest.mark.parametrize(
    "ceed_app", [{'persist_config': 'base1_'}, ], indirect=True)
async def test_change_config_read(ceed_app: CeedTestApp):
    # test_change_config_write would have set it to False
    assert not ceed_app.app.view_controller.mirror_mea
    ceed_app.app.view_controller.mirror_mea = True
    assert ceed_app.app.view_controller.mirror_mea
