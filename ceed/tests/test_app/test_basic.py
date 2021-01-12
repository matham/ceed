from ceed.tests.ceed_app import CeedTestApp
import pytest

pytestmark = pytest.mark.ceed_app


async def test_open_app(ceed_app: CeedTestApp):
    await ceed_app.wait_clock_frames(5)


async def test_app_settings(ceed_app: CeedTestApp):
    assert not ceed_app.app.ceed_data.filename
    assert ceed_app.app.ceed_data.backup_filename
    assert ceed_app.app.ceed_data.nix_file is not None
