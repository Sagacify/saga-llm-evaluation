# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LLM Evaluation Library'
copyright = '2024, Sagacify, Leonardo Remondini (leonardo.remondini@sagacify.com), Lucie Navez de Lamotte (lucie.navez@sagacify.com)'
author = 'Leonardo Remondini (leonardo.remondini@sagacify.com), Lucie Navez de Lamotte (lucie.navez@sagacify.com)'
release = '0.7.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    'sphinx_markdown_builder',
    'sphinx_emoji_favicon'
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
root_doc = "index"
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = "_static/sagacify_logo.png"
html_favicon = '_static/logo.png'
html_static_path = ['_static']
html_title = "LLM Evaluation Library"
html_css_files = [
    'css/custom.css',
]