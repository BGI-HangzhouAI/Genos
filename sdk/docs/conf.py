# Configuration file for the Sphinx documentation builder

import os
import sys
from datetime import datetime
from sphinx.ext.autodoc import between

# -- Path setup --------------------------------------------------------------
# Add the project root to sys.path so Sphinx can find modules
sys.path.insert(0, os.path.abspath('../..'))  # Assuming docs/ is inside genos/docs/

# -- Project information -----------------------------------------------------
project = 'Genos SDK'
author = 'Genos Team'
copyright = f'{datetime.now().year}, {author}'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Automatically generate documentation from docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.napoleon',      # Support for Google/NumPy style docstrings
    'sphinx.ext.todo',          # Support for TODO directives
    'sphinx.ext.autosummary',   # Generate API summary automatically
    'myst_parser',              # Support Markdown files
    'nbsphinx',                 # Support Jupyter Notebooks
]

# Recognized source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Automatically generate summary tables
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}


# Notebook execution settings
nbsphinx_execute = 'never'     # Do not execute notebook cells
nbsphinx_allow_errors = True   # Ignore execution errors

# Avoid adding module names as prefixes to class names
add_module_names = False

# Autodoc default options
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Do not add "View page source" link at the top
html_show_sourcelink = False

# Source code view settings
viewcode_follow_imported_members = False

# Enable TODOs in the documentation
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_nav_header_background': '#343131',
}
html_static_path = ['_static']

# -- Autodoc settings --------------------------------------------------------
autoclass_content = 'class'      # Show only the class docstring, not __init__
autodoc_member_order = 'bysource'  # Display members in source code order

# -- Suppress specific warnings ----------------------------------------------
suppress_warnings = [
    'toc.not_readable',
]

# -- Napoleon settings ------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Custom docstring processing --------------------------------------------
# Remove unnecessary metadata lines like @File, @Author, @Date, @Desc
def process_docstring(app, what, name, obj, options, lines):
    filtered = [
        line for line in lines
        if not line.startswith(("@File", "@Author", "@Mail", "@Date", "@Desc"))
    ]
    lines[:] = filtered

def setup(app):
    app.connect("autodoc-process-docstring", process_docstring)
