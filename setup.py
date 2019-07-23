from setuptools import setup, find_packages
import ceed

with open('README.rst') as fh:
    long_description = fh.read()

setup(
    name='Ceed',
    version=ceed.__version__,
    author='Matthew Einhorn',
    author_email='moiein2000@gmail.com',
    url='http://matham.github.io/ceed/',
    license='MIT',
    description='Slice stimulation and recordings.',
    long_description=long_description,
    classifiers=['License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering',
                 'Topic :: System :: Hardware',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Operating System :: Microsoft :: Windows',
                 'Intended Audience :: Developers'],
    packages=find_packages(),
    install_requires=[
        'ffpyplayer', 'cplcom', 'kivy', 'numpy', 'scikit-image', 'psutil',
        'nixio', 'tqdm', 'scipy', 'kivy_garden.graph',
        'kivy_garden.filebrowser', 'kivy_garden.collider', 'pytest',
        'pytest-trio', 'McsPyDataTools', 'kivy_garden.drag_n_drop'],
    dependency_links=[
        'https://kivy-garden.github.io/simple/'
    ],
    package_data={'ceed': ['data/*', '*.kv']},
    entry_points={'console_scripts': ['ceed=ceed.main:run_app']},
    )
