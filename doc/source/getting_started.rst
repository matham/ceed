Getting Started
================

Introduction
-------------

Ceed is a Python based application to help run in-vitro experiments that
optically stimulates brain slices and records their activity.

Ceed is the user interface for describing and running the experiments.
The overall system is described in :ref:`ceed-blueprint`.
See the :ref:`ceed-guide` for the Ceed specific guide.

Complete developer API documentation is at :ref:`ceed-root-api`.

Usage
------

Ceed is normally run on the PCs connected to the required hardware described
in :ref:`ceed-blueprint`. However, it can also be run in a demo-mode on
a Windows/Linux PC without any additional hardware.

To run Ceed, you first need to install it as a python package; see :ref:`install-ceed`.

After it's installed, you can run it by running the ``run_app`` file, or by entering
``ceed`` in the terminal. For example::

    python ceed/run_app.py
    # or if you are in the python enviroenment where ceed was installed, just
    ceed

A compiled executable can be downloaded from the
`release page <https://github.com/matham/ceed/releases>`__ for Windows. This can be run
as a demo, without installation.

Configuration
-------------

Ceed can be fully configured through a
`yaml file <https://github.com/matham/ceed/blob/master/ceed/data/CeedApp_config.yaml>`__.
This file is normally in the data directory under ceed, where ceed is installed.
Documentation for all the configuration options is listed in the `CEED Config
section <#CEED Config>`__.
