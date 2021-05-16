import os
import shutil
import pathlib
from docutils import nodes
from docutils.parsers.rst import directives, Directive


class video(nodes.General, nodes.Element):
    pass


class Video(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 5
    final_argument_whitespace = False
    option_spec = {
        "noloop": directives.flag,
        "nomaxsize": directives.flag,
        "alt": directives.unchanged,
        "width": directives.unchanged,
        "height": directives.unchanged,
        "noautoplay": directives.flag,
        "nocontrols": directives.flag,
    }

    def run(self):
        fname = pathlib.Path(self.arguments[0])

        noloop = self.options.get("noloop", False)
        nomaxsize = self.options.get("nomaxsize", False)
        alt = self.options.get("alt", fname.stem)
        width = self.options.get("width", "")
        height = self.options.get("height", "")
        noautoplay = self.options.get("noautoplay", False)
        nocontrols = self.options.get("nocontrols", False)

        return [video(
            path=self.arguments[0],
            alt=alt,
            width=width,
            height=height,
            noautoplay=noautoplay,
            nocontrols=nocontrols,
            noloop=noloop,
            nomaxsize=nomaxsize,
        )]


def visit_video_node(self, node):
    name = pathlib.Path(node["path"]).name
    out_dir = pathlib.Path(self.builder.confdir).joinpath('_videos')
    rel_name = os.path.relpath(out_dir.joinpath(name), pathlib.Path(node.source).parent)
    print(rel_name)
    extension = os.path.splitext(node["path"])[1][1:]

    maxsize = ''
    if not node["nomaxsize"]:
        maxsize = \
            'title="Open in tab for original size" style="max-width: 100%;">'

    html_block = '''
    <video {noloop} {width} {height} {nocontrols} {noautoplay} {maxsize}
    <source src="{rel_name}" type="video/{filetype}">
    {alt}
    </video>
    '''.format(
        noloop="" if node["noloop"] else "loop",
        width="width=\"" + node["width"] + "\"" if node["width"] else "",
        height="height=\"" + node["height"] + "\"" if node["height"] else "",
        rel_name=rel_name,
        filetype=extension,
        alt=node["alt"],
        noautoplay="" if node["noautoplay"] else "autoplay",
        nocontrols="" if node["nocontrols"] else "controls",
        maxsize=maxsize,
        )
    self.body.append(html_block)


def depart_video_node(self, node):
    pass


def copy_files(app, doctree, fromdocname):
    out_dir = os.path.join(app.builder.outdir, '_videos')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for node in doctree.traverse(video):
        fname = pathlib.Path(node.source).parent.joinpath(node['path'])
        shutil.copy2(fname, out_dir)


def setup(app):
    app.connect('doctree-resolved', copy_files)

    app.add_node(video, html=(visit_video_node, depart_video_node))
    app.add_directive("video", Video)

