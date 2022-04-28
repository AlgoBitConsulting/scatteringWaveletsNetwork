import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name             = "scatDocWaveNet",
    version          = "0.0.1",
    author           = "Dr. Markus Wagner",
    author_email     = "markuswagner@algobitconsulting.de",
    description      = ("detect and read tables with scattering wavelet networks"),
    license          = "BSD",
    keywords         = "scattering wavelet network",
    url              = "https://github.com/AlgoBitConsulting/scatteringWaveletsNetworks.git",
    packages         = ['modules'],
    long_description = read('README.md'),
    install_requires = ['tqdm', 
                        'numpy', 
                        'pandas-stubs', 
                        'beautifulsoup4',
                        'joblib',
                        'pytesseract',
                        'pdf2image',
                        'Wand',
                        'opencv-python',
                        'matplotlib',
                        'scipy', 
                        'scikit-learn'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
