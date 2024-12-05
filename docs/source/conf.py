# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ECTMetrics'
copyright = '2024, Max Kayser'
author = 'Max Kayser'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'recommonmark'
]

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = []#['_static']


# -- Options for recommonmark -------------------------------------------------
from recommonmark.transform import AutoStructify

def setup(app):
    app.add_config_value('recommonmark_config', {
        'enable_eval_rst': True,
        'auto_toc_tree_section': 'Contents',
        'auto_toc_tree_depth': 2,
        'github_user': 'maxkayser',
        'github_repo': 'ectmetrics',  
        'github_version': 'main',  # replace with your default branch
        'use_gitblit': False,
    }, True)

    app.add_transform(AutoStructify)

source_suffix = ['.rst', '.md']  # Keep this for markdown support