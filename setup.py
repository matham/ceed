from setuptools import setup, find_packages
import ceed

with open('README.rst') as fh:
    long_description = fh.read()


def get_garden(package):
    return (
        'kivy_garden.{0} @ '
        'https://github.com/kivy-garden/{0}/archive/master.zip'
        '#egg=kivy_garden.{0}'.format(package))


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
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Operating System :: Microsoft :: Windows',
                 'Intended Audience :: Developers'],
    packages=find_packages(),
    install_requires=[
        'ffpyplayer', 'base_kivy_app', 'kivy', 'numpy', 'scikit-image',
        'psutil', 'nixio', 'tqdm', 'scipy', get_garden('graph'),
        get_garden('filebrowser'), get_garden('collider'), 'pytest',
        'pytest-trio', 'McsPyDataTools', get_garden('drag_n_drop'),
        'pytest-cov', get_garden('painter'), 'trio', 'sphinx-rtd-theme',
        'cpl_media'],
    package_data={'ceed': ['data/*', '*.kv']},
    entry_points={'console_scripts': ['ceed=ceed.main:run_app']},
    )
