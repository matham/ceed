from setuptools import setup, find_packages
import ceed

with open('README.rst') as fh:
    long_description = fh.read()

setup(
    name='ceed',
    version=ceed.__version__,
    author='Matthew Einhorn',
    author_email='moiein2000@gmail.com',
    url='https://matham.github.io/ceed/',
    license='MIT',
    description='Slice stimulation and recordings.',
    long_description=long_description,
    classifiers=['License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering',
                 'Topic :: System :: Hardware',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Operating System :: Microsoft :: Windows',
                 'Intended Audience :: Developers'],
    packages=find_packages(),
    install_requires=[
        'ffpyplayer', 'base_kivy_app', 'kivy', 'numpy', 'scikit-image',
        'psutil', 'nixio==1.4.9', 'tqdm', 'scipy', 'kivy_garden.graph~=0.4.0',
        'kivy_garden.collider~=0.1.0',
        'McsPyDataTools', 'kivy_garden.drag_n_drop~=0.1.0',
        'kivy_garden.painter~=0.2.0', 'trio', 'cpl_media', 'tree-config'],
    extras_require={
        'dev': ['pytest>=3.6', 'pytest-cov', 'flake8', 'sphinx-rtd-theme',
                'coveralls', 'trio', 'pytest-trio', 'pyinstaller',
                'pytest-kivy'],
    },
    package_data={'ceed': ['data/*', '*.kv']},
    entry_points={'console_scripts': ['ceed=ceed.main:run_app']},
)
