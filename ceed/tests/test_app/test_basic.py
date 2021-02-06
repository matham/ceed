import pytest
import pathlib

import ceed
from ceed.tests.ceed_app import CeedTestApp
from .examples import create_test_image, assert_image_same

pytestmark = pytest.mark.ceed_app


async def test_open_app(ceed_app: CeedTestApp):
    await ceed_app.wait_clock_frames(5)


async def test_app_settings(ceed_app: CeedTestApp):
    assert not ceed_app.app.ceed_data.filename
    assert ceed_app.app.ceed_data.backup_filename
    assert ceed_app.app.ceed_data.nix_file is not None


async def test_app_save_load_screenshot(ceed_app: CeedTestApp, tmp_path):
    assert ceed_app.app.player.last_image is None

    filename = str(tmp_path / 'image.bmp')
    image = create_test_image(512, 234)
    ceed_app.app.player.save_screenshot(image, ceed_app.app, [filename])

    assert ceed_app.app.player.last_image is None

    ceed_app.app.player.load_screenshot(ceed_app.app, [filename])
    assert ceed_app.app.player.last_image is not None
    assert_image_same(image, ceed_app.app.player.last_image)


async def test_app_save_analysis_image(ceed_app: CeedTestApp, tmp_path):
    from ceed.analysis import CeedDataReader
    reader = CeedDataReader('')
    assert ceed_app.app.player.last_image is None

    filename = str(tmp_path / 'image.bmp')
    image = create_test_image(342, 435)
    reader.save_image(filename, image)

    assert ceed_app.app.player.last_image is None
    ceed_app.app.player.load_screenshot(ceed_app.app, [filename])
    assert ceed_app.app.player.last_image is not None
    assert_image_same(image, ceed_app.app.player.last_image)


async def test_app_load_h5_image(ceed_app: CeedTestApp, tmp_path):
    root = pathlib.Path(ceed.__file__).parent.joinpath('examples', 'data')
    filename = str(root.joinpath('ceed_data.h5'))
    image = create_test_image(250, 500)

    assert ceed_app.app.player.last_image is None
    ceed_app.app.player.load_screenshot(ceed_app.app, [filename])
    assert ceed_app.app.player.last_image is not None
    assert_image_same(image, ceed_app.app.player.last_image, exact=False)
