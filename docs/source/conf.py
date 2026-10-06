# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))




# -- Project information

project = 'ADAlib'
copyright = '2026, Youngwoo Yoo, Dayeong Kang, Suhyeong Lim, Sang-Hyun Rhie, Jeongsu Lee'
author = 'Youngwoo Yoo, Dayeong Kang, Suhyeong Lim, Sang-Hyun Rhie, Jeongsu Lee'

release = '0.1.0'
version = '0.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

autosummary_generate = True
autodoc_member_order = "bysource"

autodoc_mock_imports = [
    "tensorflow",
    "torch",
    "jax",
    "deepxde",
    "scipy",
    "numpy",
    "matplotlib",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
