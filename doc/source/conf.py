# -*- coding: utf-8 -*-

from functools import partial

import ceed
import kivy
from ceed.main import CeedApp
from cplcom.config import create_doc_listener, write_config_attrs_rst

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx'
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

html_theme_options = {
    'github_button': 'true',
    'github_banner': 'true',
    'github_user': 'matham',
    'github_repo': 'ceed'
}

intersphinx_mapping = {
    'pyflycap2': ('https://matham.github.io/pyflycap2/', None),
    'ffpyplayer': ('https://matham.github.io/ffpyplayer/', None),
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
html_theme = 'alabaster'

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
    create_doc_listener(app, ceed)
    app.connect(
        'build-finished',
        partial(write_config_attrs_rst, CeedApp.get_config_classes(),
                ceed)
    )
