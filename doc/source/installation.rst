.. _install-ceed:

*************
Installation
*************

Ceed can be found at https://github.com/matham/ceed.

Dependencies
-------------

Following are the overall kivy dependencies. For exact instructions on how to install
them see below.

* Python 3.6+.
* Kivy 2.0.0+ and Kivy-garden widgets for the GUI.
* numpy/scipy for the scientific stack.
* nixio to read/write data files.
* pypixxlib to control the PROPixx projector.
* McsPyDataTools to read the MCS generated e-phys data files.
* pyflycap2 when using Point Gray cameras.
* thorcam when using Thor cameras.
* ffpyplayer for media control.
* base_kivy_app provides a basic Kivy app structure.
* cpl_media provides control over the various cameras.


Installation on Ubuntu
----------------------

Following is a step by step example installation of Ceed on Ubunutu 18.04, specifically
tested on the computer that is used to run Ceed for the experiments.

Install the apt dependencies
****************************

* ``sudo apt update``.
* ``sudo apt install python3 python3-dev python3-pip``.

Make a virtual env for the project
**********************************

We'll make the virtual env in home:

* Install update pip/virtualenv ``python3 -m pip install --upgrade --user pip virtualenv``.
* ``cd ~``.
* Make the virtualenv ``python3 -m virtualenv ceed_venv``.
* Activate it ``source ceed_venv/bin/activate``. You'll have to do this every time you start a new terminal.

Install manual python dependencies
**********************************

* Install current **kivy** master

  * Install apt dependencies: ``sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev zlib1g-dev``.
  * Install kivy: ``pip install https://github.com/kivy/kivy/archive/master.zip`` - it'll take a couple of minutes.
* If using a **PointGray camera**, on linux we must manually install its libraries

  * Get it from `here <https://www.flir.com/products/flycapture-sdk>`_, extract it and install by running ``install_flycapture.sh``.
  * Figure out your python version, find the appropriate linux wheel of the last release
    `here <https://github.com/matham/pyflycap2/releases>`_ and install e.g. with
    ``pip install https://github.com/matham/pyflycap2/releases/download/v0.3.0/pyflycap2-0.3.0-cp36-cp36m-linux_x86_64.whl``.
  * If successful, you should be able to run
    ``python3 -c 'from pyflycap2.interface import CameraContext; cc = CameraContext(); cc.rescan_bus(); print(cc.get_gige_cams())'``
    and it'll print a list of the serial numbers of all the connected cameras.
* Install **VPixx** control software as well as ``pypixxlib``

  * Go to their `site <https://vpixx.com/>`_ and download the vpixx debian package for ubuntu named e.g. ``vpixx-software-tools.deb``.
  * Install it with ``sudo apt install ./vpixx-software-tools.deb`` from the download directory.
  * To install ``pypixxlib`` you'll need to locate ``pypixxlib-xxx.tar.gz``, likely under ``/usr/share/VPixx Software Tools/Software Tools/pypixxlib``
    and install it with e.g. ``pip install "/usr/share/VPixx Software Tools/Software Tools/pypixxlib/pypixxlib-3.5.5601.tar.gz"``.
  * To test if it is installed successfully, run ``python3 -c "from pypixxlib import _libdpx as libdpx"``.

    * If you get an error about print, the library doesn't support python3 yet and you'll need to fix it as follows:
    * Locate where the library was installed by running ``python3 -c "import pypixxlib; print(pypixxlib.__path__)"``.
      This should print something like ``['/home/cpl/ceed_venv/lib/python3.6/site-packages/pypixxlib']``.
    * cd to that directory with e.g. ``cd /home/cpl/ceed_venv/lib/python3.6/site-packages/pypixxlib``.
    * Run ``2to3 -w .`` to convert any python 2 code to python 3 as best as possible. Ignore all the printed output.
    * Test again with ``python3 -c "from pypixxlib import _libdpx as libdpx"`` and it shouldn't raise any issues anymore.

Install Ceed
************

Manually clone and install **ceed** and associated projects.
We'll clone it into PycharmProjects: ``cd  ~/PycharmProjects/``
Fow now, while the code is still changing, we'll also clone ``base_kivy_app`` and ``cpl_media``
and install them in place, rather than installing them like a normal pip dependency.
Consequently, we'll be able to pull the changes easily.

* Install ``base_kivy_app``

  * Clone with ``git clone https://github.com/matham/base_kivy_app.git``.
  * Install in place with ``pip install -e base_kivy_app``.
* Install ``cpl_media``

  * Clone with ``git clone https://github.com/matham/cpl_media.git``.
  * Install in place with ``pip install -e cpl_media``.
* Install ``ceed`` finally

  * Clone with ``git clone https://github.com/matham/ceed.git``.
  * Install in place with ``pip install -e ceed``.

Once installed, you can start Ceed by simply typing ``ceed`` in the terminal.
Or, you can run it directly using ``python -m ceed.run_app``. Or from the
ceed directory, just run ``python ceed/run_app.py``.

Installation on Windows
-----------------------

Following is a step by step example installation of Ceed on Windows 10, specifically
tested on the computer that runs the camera and MCS.

Make a virtual env for the project
**********************************

Starting with Python and git available on the terminal, we'll first make the virtual env in the home
directory. The terminal should be in the home directory

* Install update pip/virtualenv ``python -m pip install --upgrade pip virtualenv``.
* Make the virtualenv ``python -m virtualenv ceed_venv``.
* Activate it ``ceed_venv\Scripts\activate``. You'll have to do this every time you start a new terminal.

Install manual python dependencies
**********************************

* Install current **kivy** master with ``pip install kivy[base] kivy_examples --pre --extra-index-url https://kivy.org/downloads/simple/``.
* If using a **PointGray camera** install with ``pip install pyflycap2``.

  * If successful, you should be able to run
    ``python -c "from pyflycap2.interface import CameraContext; cc = CameraContext(); cc.rescan_bus(); print(cc.get_gige_cams())"``
    and it'll print a list of the serial numbers of all the connected cameras.
* If using a **Thor camera** install with ``pip install thorcam``.
* Install **VPixx** control software as well as ``pypixxlib``. Although we're not typically running ceed on this computer,
  it can be tested here.

  * Go to their `site <https://vpixx.com/>`_ and download the vpixx Windows executable for Windows named e.g.
    ``setup.exe`` and install it.
  * To install ``pypixxlib`` you'll need to locate ``pypixxlib-xxx.tar.gz``, likely under ``C:\Program Files\VPixx Technologies\Software Tools\pypixxlib``
    and install it with e.g. ``pip install "C:\Program Files\VPixx Technologies\Software Tools\pypixxlib\pypixxlib-3.5.5428.tar.gz"``.
  * To test if it is installed successfully, run ``python -c "from pypixxlib import _libdpx as libdpx"``.

    * If you get an error about print, the library doesn't support python3 yet and you'll need to fix it as follows
    * Locate where the library was installed by running ``python -c "import pypixxlib; print(pypixxlib.__path__)"``.
      This should print something like ``['C:\\Users\\MEArig\\ceed_venv\\lib\\site-packages\\pypixxlib']``.
    * cd to that directory with e.g. ``cd C:\Users\MEArig\ceed_venv\Lib\site-packages\pypixxlib``.
    * Install ``2to3`` with ``pip install 2to3 ``.
    * Run ``2to3 -w .`` to convert any python 2 code to python 3 as best as possible. Ignore all the printed output.
    * Test again with ``python -c "from pypixxlib import _libdpx as libdpx"`` and it shouldn't raise any issues anymore.

Install Ceed
************

Manually clone and install **ceed** and associated projects.
We'll clone it into PycharmProjects: ``cd  C:\Users\MEArig\PycharmProjects``
Fow now, while the code is still changing, we'll also clone ``base_kivy_app`` and ``cpl_media``
and install them in place, rather than installing them like a normal pip dependency.
Consequently, we'll be able to pull the changes easily.

* Install ``base_kivy_app``

  * Clone with ``git clone https://github.com/matham/base_kivy_app.git``.
  * Install in place with ``pip install -e base_kivy_app``.
* Install ``cpl_media``

  * Clone with ``git clone https://github.com/matham/cpl_media.git``.
  * Install in place with ``pip install -e cpl_media``.
* Install ``ceed`` finally

  * Clone with ``git clone https://github.com/matham/ceed.git``.
  * Install in place with ``pip install -e ceed``.

Once installed, you can start Ceed by simply typing ``ceed`` in the terminal.
Or, you can run it directly using ``python -m ceed.run_app``. Or from the
ceed directory, just run ``python ceed/run_app.py``.

Similarly, to run the cameras using the ``cpl_media`` app, just type ``cpl_media`` in the terminal.

Using it with PyCharm
---------------------

To use the installed projects from PyCharm, in PyCharm create a "new
project", point to the ceed directory and create it from the existing
folder.

When selecting a python installation, make sure to point to the virtual env
and use that for all the projects.
Do similarly for ``base_kivy_app`` and ``cpl_media`` if you want to access it
from PyCharm.
