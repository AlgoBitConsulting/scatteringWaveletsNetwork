
import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name             = "docScatWaveNet",
    version          = "0.0.1",
    author           = "Dr. Markus Wagner",
    author_email     = "markuswagner@algobitconsulting.de",
    description      = ("detect and read tables with scattering wavelet networks"),
    license          = "BSD",
    keywords         = "scattering wavelet network",
    url              = "https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    long_description = read('README.md'),
    install_requires = ['tqdm', 
                        'numpy', 
                        'pandas-stubs', 
                        'beautifulsoup4',
                        'joblib',
                        'pytesseract',
                        'pdf2image',
                        'wand',
                        'opencv-python',
                        'matplotlib',
                        'scipy', 
                        'scikit-learn'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
   
)

