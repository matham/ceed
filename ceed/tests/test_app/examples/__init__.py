from ffpyplayer.pic import Image


def create_test_image(width: int = 100, height: int = 100) -> Image:
    size = width * height * 3
    buf = bytearray([int(x * 255 / size) for x in range(size)])
    img = Image(plane_buffers=[buf], pix_fmt='bgr24', size=(width, height))

    return img


def assert_image_same(image1: Image, image2: Image, exact=True) -> None:
    assert image1.get_pixel_format() == image2.get_pixel_format()
    assert image1.get_size() == image2.get_size()
    assert image1.get_buffer_size() == image2.get_buffer_size()

    data1 = image1.to_bytearray()
    data2 = image2.to_bytearray()
    assert data1[1:] == data2[1:]
    if exact:
        assert data1[0] == data2[0]
    else:
        for a, b in zip(data1[0], data2[0]):
            assert a - 1 <= b <= a + 1
