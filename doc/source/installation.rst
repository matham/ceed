.. _install-ceed:

Installation
============

`Ceed <https://github.com/matham/ceed>`__ requires Python 3.7+.

.. _demo-install:

Demo install
------------

For Windows you can download a pre-compiled Ceed executable (``Ceed.exe``) from the
`release page <https://github.com/matham/ceed/releases>`_.

Alternatively, the most recent stable version of Ceed can be installed from
source simply with::

    pip install ceed
    # or if you want to install the most recent pre-release version
    pip install ceed --pre

This is sufficient to run Ceed as a demo on Windows/Linux without the connected hardware
of the "real" system. See :ref:`usage` for how to run Ceed.

.. note::

    When running Ceed installed as above, you'll likely get the following error logged to
    the log window: ``ImportError: Cannot open PROPixx library`` when starting Ceed.
    This is normal and happens because the projector control software is not installed.
    It can be safely ignored for demo purposes.

    Also, if the Teensy microcontroller is not connected but "Use Teensy" is enabled in the
    :ref:`tut-config-window`, Ceed won't run experiments. Turn it OFF for a demo install.


Installation on Ubuntu
----------------------

The following sets up Ceed and the computer to be able to run actual
experiments and control the required external hardware. It was specifically
tested on the Ubuntu 20.04 computer that is used to run Ceed for the experiments.

Install the system dependencies
*******************************

::

    sudo apt update
    sudo apt install python3-pip build-essential git python3 python3-dev libusb-1.0-0
    sudo apt-get install -y \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-mixer-dev \
        libsdl2-ttf-dev \
        libportmidi-dev \
        libswscale-dev \
        libavformat-dev \
        libavcodec-dev \
        zlib1g-dev \
        xclip

Once the system is updated, check in the additional drivers that the correct Nvidia drivers are installed.
Then verify by starting nvidia-server and inspect that the frame-rate of the GPU is 119.96.

Also, turn OFF the screensaver and set notifications to OFF to prevent any interruptions during the experiment.

.. _install-venv-linux:

Make a virtual env for the project
**********************************

We'll make the virtual env in home. This will self-contain all the packages needed for Ceed:

* Install update pip/virtualenv ``python3 -m pip install --upgrade --user pip virtualenv``.
* ``cd ~``.
* Make the virtualenv ``python3 -m virtualenv venv_ceed``.
* Activate it ``source venv_ceed/bin/activate``. You'll have to do this every time you start a new terminal.

.. _install-dep-linux:

Install manual python dependencies
**********************************

* Install **kivy** from source. We cannot use a pre-compiled kivy from pip because it doesn't work as well in
  full-screen.
  * Install kivy: ``python3 -m pip install kivy[full]==2.0.0 --no-binary kivy`` - it'll take a couple of minutes.
* If using a **PointGray camera**, on linux we must manually install its libraries

  * Get it from `here <https://www.flir.com/products/flycapture-sdk>`__, extract it and install by running ``install_flycapture.sh``.
  * Figure out your python version, find the appropriate linux wheel of the last release
    `here <https://github.com/matham/pyflycap2/releases>`__ and install e.g. with
    ``pip install https://github.com/matham/pyflycap2/releases/download/v0.3.0/pyflycap2-0.3.0-cp36-cp36m-linux_x86_64.whl``.
  * If successful, you should be able to run
    ``python3 -c 'from pyflycap2.interface import CameraContext; cc = CameraContext(); cc.rescan_bus(); print(cc.get_gige_cams())'``
    and it'll print a list of the serial numbers of all the connected cameras.
* Install **VPixx** control software as well as ``pypixxlib``

  * Go to their `site <https://vpixx.com/>`__ and download the vpixx debian package for ubuntu named e.g. ``vpixx-software-tools.deb``.
  * Install it with ``sudo apt install ./vpixx-software-tools.deb`` from the download directory.
  * To install ``pypixxlib`` you'll need to locate ``pypixxlib-xxx.tar.gz``, likely under ``/usr/share/VPixx Software Tools/Software Tools/pypixxlib``
    and install it with e.g. ``pip install "/usr/share/VPixx Software Tools/Software Tools/pypixxlib/pypixxlib-3.8.9279.tar.gz"``.
  * To test if it is installed successfully, run ``python3 -c "from pypixxlib import _libdpx as libdpx"``.
  * Using the vputil program installed with the vpixx package, from the command line update the projector and controller
    firmware (they are named ppx and ppc).

.. _install-ceed-linux:

Install Ceed
************

Manually clone and install **Ceed**.
We'll clone it into PycharmProjects (assuming this folder exists, otherwise create it)::

    cd  ~/PycharmProjects/
    git clone https://github.com/matham/ceed.git
    cd ceed
    python3 -m pip install -e .[dev]

Once installed, you can start Ceed by simply typing ``ceed`` in the terminal.
Or, you can run it directly from the ceed directory with ``python3 ceed/run_app.py``.

To use the installed projects from PyCharm, in PyCharm create a "new
project", point to the ceed directory and create it from the existing
folder. When selecting a python installation, make sure to select/add the
virtual environment Python previously created.

The first time running Ceed, make sure the LED idle mode is set to ``none`` and that the frame rate
is set to ``119.96``.

Teensy setup
************

Before the Teensy can be used by Ceed, the USB device must be correctly set up.
Do the following::

    # in the terminal start nano with the file to be created.
    sudo nano /etc/udev/rules.d/10-local.rules
    # paste the following line into the file
    ACTION=="add", SUBSYSTEMS=="usb", ATTR{idVendor}=="16c0", ATTR{idProduct}=="0486", MODE="660", GROUP="plugdev"
    # now save the file and back in the terminal run the following
    sudo adduser $USER plugdev

To program the Teensy if it's a new device, please follow the `instructions in the repo
<https://github.com/matham/ceed/blob/master/ceed/view/teensy_estimation/readme.md>`__.

.. _linux-network:

Network setup
*************

In order to send the large camera images quickly between the Windows to Ubuntu PC, we need to increase the
packet sizes of the Network. First list the network devices with ``ip link show``. This will print something like::

    2: enp0s31f6: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP mode DEFAULT group default qlen 1000
        link/ether ...
    3: wlp5s0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN mode DORMANT group default qlen 1000
        link/ether ...

Notice that ``MTU`` is set to 1500. We need to increase it to the max, 9000 as follows. In the terminal
run ``sudo nano /etc/init.d/cam_mtu.sh``. This will open the nano text editor with the given file. In there paste in the
following (change the network name as needed)::

    ip link set enp0s31f6 mtu 9000
    ip link set wlp5s0 mtu 9000

Restart the computer and check that the ``MTU`` is 9000.

Shared drive
~~~~~~~~~~~~
We also need to share a directory over the network between the Windows and Ubuntu computer. Once the appropriate
directory on Windows was set to be shared (e.g. ``D:\MC_Rack data``), from e.g. the login screen
get the Windows computer's name and username. In Ubuntu in Files, under "connect to server" enter smb://computer_name
and when prompted enter the Windows username, workgroup name and password to connect. Select to remember pass forever.
Then find the shared folder and right-click -> mount to show it.

Camera streaming
~~~~~~~~~~~~~~~~

When you have Filers configured on the Windows computer, in Ceed you have to select the appropriate IP and port to be
able to stream the camera images from Windows to it. In the player, select network, then enter the IP of the
Windows computer and use 10000 for the port. It should be able to connect if the Filers server is running.

Installation on Windows
-----------------------

Following installs Ceed on the Windows computer that runs the camera and MCS. It's not strictly needed, but can be used
for testing if desired. However, the Filers and MCS step are necessary to be able to run experiments.

First ensure that power mode is set to never put the computer to sleep and that the screensaver is OFF.

.. _install-venv-win:

Make a virtual env for the project
**********************************

Starting with Python (install it if needed) available on the terminal, we'll first make the virtual env
in the home directory. The terminal should be in the home directory

* Install update pip/virtualenv ``python -m pip install --upgrade pip virtualenv``.
* Make the virtualenv ``python -m virtualenv ceed_venv``.
* Activate it ``ceed_venv\Scripts\activate``. You'll have to do this every time you start a new terminal.

.. _install-dep-win:

Install manual python dependencies
**********************************

* If using a **PointGray camera** install with ``pip install pyflycap2``.

  * If successful, you should be able to run
    ``python -c "from pyflycap2.interface import CameraContext; cc = CameraContext(); cc.rescan_bus(); print(cc.get_gige_cams())"``
    and it'll print a list of the serial numbers of all the connected cameras.
* If using a **Thor camera** install with ``pip install thorcam``.
* Install **VPixx** control software as well as ``pypixxlib``. Although we're not typically running ceed on this computer,
  it can be tested here.

  * Go to their `site <https://vpixx.com/>`__ and download the vpixx Windows executable for Windows named e.g.
    ``setup.exe`` and install it.
  * To install ``pypixxlib`` you'll need to locate ``pypixxlib-xxx.tar.gz``, likely under ``C:\Program Files\VPixx Technologies\Software Tools\pypixxlib``
    and install it with e.g. ``pip install "C:\Program Files\VPixx Technologies\Software Tools\pypixxlib\pypixxlib-3.5.5428.tar.gz"``.
  * To test if it is installed successfully, run ``python -c "from pypixxlib import _libdpx as libdpx"``.

.. _install-ceed-win:

Install Ceed
************

Ceed can simply be installed with::

    pip install ceed[dev]

Once installed, you can start Ceed by simply typing ``ceed`` in the terminal.
Or, you can run it directly using ``python ceed/run_app.py``.

You can alternatively clone it and install it in-place like in the Ubuntu instructions.

.. _win-filers:

Thor/Filers
***********

To be able to play the images from the Thor camera, first download and `install the Thor drivers
<https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`__. Then power the
camera and ensure it works in the Thor camera software.

Next, download the `Filers <https://github.com/matham/filers2/releases>`__ exe, run and pin to taskbar.
Within filers, with the camera powered, select the thor as the player and Network as the recorder.
In the recorder settings enter the ip address of the Windows computer and use 10000 for the port.
To test start playing, start the server and press the record button to stream to the network.
On the Ubuntu computer similarly connect to the server from Ceed and you should see the camera images
in Ceed.

To be able to efficiently stream the images, in device manager for all the network adapters used,
locate advanced settings and set Jumbo packet (possibly listed as MTU) value to 9014 bytes, the maximum.

MCS
****

To control the MCS hardware, ensure the following MCS software are installed and updated: MC Experimenter,
MC Data manager, and MC Analyzer. Then, in Experimenter, double clock on the MEA (while it's powered)
and update all firmware.


Ceed-MCS hardware link
----------------------

Once the projector and controllers are all connected to the appropriate computers, we must use the DB-to-BNC cable to
connect the Projector controller corner pixel port to the MCS digital input. In the ceed configuration file
locate the ``projector_to_aquisition_map`` setting. That indicates the mapping from vpixx port to MCS port.
E.g. ``2: 0`` means vpixx bit/port 2 should be connected to the port 0 of the MCS digital input breakout box.
