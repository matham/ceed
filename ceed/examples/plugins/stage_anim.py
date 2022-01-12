"""This plugin shows how to create custom graphics to display during a
stage and how log RGBA and arbitrary data to the HDF5 file.
"""
from ceed.stage import CeedStage, StageDoneException
import numpy as np
from typing import Optional, Dict


# this shader is used to generate a time varying plasma image - it's very
# low-level GL
plasma_shader = '''
$HEADER$

uniform vec2 resolution;
uniform float time;

void main(void)
{
   vec4 frag_coord = frag_modelview_mat * gl_FragCoord;
   float x = frag_coord.x;
   float y = frag_coord.y;
   float mov0 = x+y+cos(sin(time)*2.)*100.+sin(x/100.)*1000.;
   float mov1 = y / resolution.y / 0.2 + time;
   float mov2 = x / resolution.x / 0.2;
   float c1 = abs(sin(mov1+time)/2.+mov2/2.-mov1-mov2+time);
   float c2 = abs(sin(c1+sin(mov0/1000.+time)
              +sin(y/40.+time)+sin((x+y)/100.)*3.));
   float c3 = abs(sin(c2+cos(mov1+mov2+c2)+cos(mov2)+sin(x/1000.)));
   gl_FragColor = vec4( c1,c2,c3,1.0);
}
'''


class PlasmaStage(CeedStage):
    """Displays a time varying plasma as an experiment stage.
    """

    gl_ctx: dict = {}
    """The gl context used to run the shader, one for each quad if in quad mode, 
    otherwise just 1.
    """

    last_time = 0.
    """The last time value passed by ceed to the stage in tick. Used to update
    the plasma in ``set_gl_colors``.
    """

    def __init__(self, **kwargs):
        # set a custom name for the stage class
        self.name = 'Plasma'
        super().__init__(**kwargs)
        # we can't pre-compute this stage because of how we get the time used
        # during drawing
        self.disable_pre_compute = True
        self.gl_ctx = {}

    def add_gl_to_canvas(
            self, screen_width: int, screen_height: int, canvas,
            name: str, quad_mode: str, quad: Optional[int] = None, **kwargs
    ) -> bool:
        # todo: this stage does not support quad12x, only quad4x
        # we overwrite this method to manually add gl graphics elements to the
        # kivy canvas
        from kivy.graphics import Color, Rectangle, RenderContext

        # create a custom canvas context, so that we can use the gl shader
        self.gl_ctx[quad] = ctx = RenderContext(
            use_parent_projection=True, use_parent_modelview=True,
            use_parent_frag_modelview=True, fs=plasma_shader, group=name)
        # pass the variables required by the shader
        ctx['time'] = 0.
        ctx['resolution'] = [screen_width / 2, screen_height / 2]

        # make sure the background behind the canvas is not transparent
        with ctx:
            Color(1, 1, 1, 1)
            Rectangle(size=(screen_width // 2, screen_height // 2))

        # add it to the overall canvas
        canvas.add(ctx)

        # must return True so that set_gl_colors is called
        return True

    def set_gl_colors(
            self, quad: Optional[int] = None, grayscale: str = None,
            clear: bool = False, **kwargs) -> None:
        # this is called for every time step, after ticking the stage when the
        # time is saved. Now pass on to the shader the current time
        self.gl_ctx[quad]['time'] = self.last_time
        if clear:
            # todo: set to black if cleared - it is not supported by plasma
            pass

    def remove_gl_from_canvas(self, *args, **kwargs) -> None:
        # everything is removed by name, so just clear the context
        self.gl_ctx = {}
        return super().remove_gl_from_canvas(*args, **kwargs)

    def evaluate_stage(self, shapes, last_end_t):
        # always get the first time
        self.t_start = t = yield
        for _ in range(self.loop):
            t_start = t

            # only go for 10 second
            while t - t_start < 10:
                # save the current time for use by the shader when
                # set_gl_colors is called after each tick
                self.last_time = float(t - t_start)

                # save some value to the shapes log - could be any 4 numbers
                shapes['plasma'].append((.5, 0, 0, 0))
                # this yields so GUI can draw shapes and resume for next frame
                t = yield

        # this time value was not used so it ends on the last sample so
        # that last time will be used as start of next stage and MUST be saved
        # as t_end
        self.t_end = t
        # this is how we indicate we're done
        raise StageDoneException

    def get_stage_shape_names(self):
        # add a "shape" named plasma so we can log data for it at every
        # timestep. The values will also display in the stage preview graph
        names = super().get_stage_shape_names()
        names.add('plasma')
        return names


class SoftEllipseStage(CeedStage):
    """Displays a time varying ellipse as an experiment stage. The ellipse is
    also zero in the center, with increased intensity as we go out from center
    """

    texture_pat: np.ndarray = None
    """A numpy pattern that will be blitted into the kivy gl texture. This
    contains the ellipse pattern. This pattern is multiplied by an attenuation
    factor at each step and copied to ``buffer`` to set the actual intensity.
    """

    buffer: Dict[int, np.ndarray] = []
    """The RGB buffer that is blitted to the texture.
    """

    buffer_count = 0
    """The current time iteration used to set the ellipse intensity.
    """

    kivy_tex: dict = {}
    """The Kivy texture, one for each quad if in quad mode, otherwise just 1.
    """

    def __init__(self, **kwargs):
        # set a custom name for the stage class
        self.name = 'SoftEllipse'
        super().__init__(**kwargs)
        # we can't pre-compute this stage because of how we get the time used
        # during drawing
        self.disable_pre_compute = True

        # generate the ellipse pattern and the RGB buffer
        x = np.linspace(-1, 1, 150)
        y = np.linspace(-1, 1, 100)
        xx, yy = np.meshgrid(x, y)
        zz = np.sqrt(np.power(xx, 2) + np.power(yy, 2))
        zz[zz > 1] = 0

        self.texture_pat = np.asarray(zz * 255, dtype=np.uint8)
        self.buffer = {}
        self.kivy_tex = {}

    def init_stage(self, *args, **kwargs) -> None:
        # reset counter to zero for ellipse intensity at the start of the stage
        self.buffer_count = 0
        return super().init_stage(*args, **kwargs)

    def init_loop_iteration(self, *args, **kwargs) -> None:
        # reset counter to zero for ellipse intensity at the start of each loop
        self.buffer_count = 0
        return super().init_loop_iteration(*args, **kwargs)

    def draw_ellipse_tex(self, quad, grayscale=None, clear=False):
        # computes the ellipse intensity using the buffer_count value computed
        # in the last tick (evaluate_stage). It then blits it
        buffer = self.buffer[quad]
        pat = np.asarray(
            self.texture_pat * (self.buffer_count / 100), dtype=np.uint8)
        if clear:
            pat = 0

        if grayscale is None:
            # we're either in normal or quad4x mode, so set the selected rgb
            # values, depending on which colors user selected in GUI
            if self.color_r:
                buffer[:, :, 0] = pat
            if self.color_g:
                buffer[:, :, 1] = pat
            if self.color_b:
                buffer[:, :, 2] = pat
            self.kivy_tex[quad].blit_buffer(buffer.reshape(-1))
        else:
            # if in quad12x, set only the specific color channel of this tick,
            # this method is called 12 times, one for each quad and channel
            channel = {'r': 0, 'g': 1, 'b': 2}[grayscale]
            buffer[:, :, channel] = pat
            if grayscale == 'b':
                # only blit once all 3 channels were updated
                self.kivy_tex[quad].blit_buffer(buffer.reshape(-1))

    def add_gl_to_canvas(
            self, screen_width: int, screen_height: int, canvas,
            name: str, quad_mode: str, quad: Optional[int] = None, **kwargs
    ) -> bool:
        # we overwrite this method to manually add gl graphics elements to the
        # kivy canvas
        from kivy.graphics import Color, Rectangle
        from kivy.graphics.texture import Texture
        with canvas:
            Color(1, 1, 1, 1, group=name)
            rect1 = Rectangle(size=(150, 100), pos=(500, 500), group=name)
            rect2 = Rectangle(size=(150, 100), pos=(1500, 500), group=name)

        tex = Texture.create(size=(150, 100), colorfmt='rgb')
        rect1.texture = tex
        rect2.texture = tex
        self.kivy_tex[quad] = tex
        self.buffer[quad] = np.zeros((100, 150, 3), dtype=np.uint8)

        # draw initial ellipse
        self.draw_ellipse_tex(quad)

        # must return True so that set_gl_colors is called
        return True

    def set_gl_colors(
            self, quad: Optional[int] = None, grayscale: str = None,
            clear: bool = False, **kwargs) -> None:
        # this is called for every time step, after ticking the stage when the
        # time is saved (i.e. buffer_count is updated). Update the ellipse
        self.draw_ellipse_tex(quad, grayscale, clear)

    def remove_gl_from_canvas(self, *args, **kwargs) -> None:
        # everything is removed by name, so just clear the texture
        self.kivy_tex = {}
        self.buffer = {}
        return super().remove_gl_from_canvas(*args, **kwargs)

    def evaluate_stage(self, shapes, last_end_t):
        r = self.color_r
        g = self.color_g
        b = self.color_b

        # always get the first time
        self.t_start = t = yield
        for _ in range(self.loop):
            t_start = t

            # only go for 10 second
            while t - t_start < 10:
                # update the counter at each tick. The ellipse is only updated
                # as well if the frame is rendered and not dropped. But it is
                # logged for every frame
                self.buffer_count = (self.buffer_count + 1) % 100

                intensity = self.buffer_count / 100
                shapes['soft_ellipse'].append((
                    intensity if r else 0.,
                    intensity if g else 0.,
                    intensity if b else 0.,
                    1
                ))
                # this yields so GUI can draw shapes and resume for next frame
                t = yield

        # this time value was not used so it ends on the last sample so
        # that last time will be used as start of next stage and MUST be saved
        # as t_end
        self.t_end = t
        # this is how we indicate we're done
        raise StageDoneException

    def get_stage_shape_names(self):
        # add a "shape" named soft_ellipse so we can log data for it at every
        # timestep. The values will also display in the stage preview graph
        names = super().get_stage_shape_names()
        names.add('soft_ellipse')
        return names


def get_ceed_stages(stage_factory):
    # return all the stage classes
    return [PlasmaStage, SoftEllipseStage]
