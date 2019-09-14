# -*- coding: utf-8 -*-

from functools import partial
import sphinx_rtd_theme
import os
import kivy  # this sets the doc include env variable
import ceed
from ceed.main import CeedApp
from cplcom.config import create_doc_listener, write_config_attrs_rst

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    "sphinx_rtd_theme",
]

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'sourcelink.html'
    ]
}

intersphinx_mapping = {
    'pyflycap2': ('https://matham.github.io/pyflycap2/', None),
    'ffpyplayer': ('https://matham.github.io/ffpyplayer/', None),
    'kivy': ('https://kivy.org/docs/', None),
    'kivy_garden.drag_n_drop':
        ('https://kivy-garden.github.io/drag_n_drop/', None),
    'kivy_garden.painter': ('https://kivy-garden.github.io/painter/', None),
    'cplcom': ('https://matham.github.io/cplcom/', None)
}

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Ceed'

# The short X.Y version.
version = ceed.__version__
# The full version, including alpha/beta/rc tags.
release = ceed.__version__

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Output file base name for HTML help builder.
htmlhelp_basename = 'Ceeddoc'

latex_elements = {}

latex_documents = [
  ('index', 'Ceed.tex', u'Ceed Documentation',
   u'Matthew Einhorn', 'manual'),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'Ceed', u'Ceed Documentation',
     [u'Matthew Einhorn'], 1)
]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'Ceed', u'Ceed Documentation',
   u'Matthew Einhorn', 'Ceed', 'One line description of project.',
   'Miscellaneous'),
]


def setup(app):
    fname = os.environ.get('CPLCOM_CONFIG_DOC_PATH', 'config_attrs.json')
    create_doc_listener(app, ceed, fname)
    if CeedApp.get_running_app() is not None:
        classes = CeedApp.get_running_app().get_app_config_classes()
    else:
        classes = CeedApp.get_config_classes()

    app.connect(
        'build-finished', partial(
            write_config_attrs_rst, classes, ceed, filename=fname)
    )
