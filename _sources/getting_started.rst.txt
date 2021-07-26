Getting Started
================

Introduction
-------------

Ceed is a Python based application to help run in-vitro experiments that
optically stimulates brain slices and records their activity.

Ceed is the user interface for describing and running the experiments.
The overall system is described in :ref:`ceed-blueprint`.
See the :ref:`ceed-guide` for a user guide.

Complete developer API documentation is at :ref:`ceed-root-api`.

.. _usage:

Usage
-----

Ceed is normally run on the PC connected to the required hardware described
in :ref:`ceed-blueprint`. However, it can also be run in a demo-mode on
a Windows/Linux PC without any additional hardware (see :ref:`demo-install`).

To run Ceed, you first need to install it as a python package; see :ref:`install-ceed`.

After it's installed, you can run it by running the ``run_app`` file, or by entering
``ceed`` in the terminal. For example::

    python ceed/run_app.py
    # or if you are in the python environment where ceed was installed, just
    ceed

Configuration
-------------

Ceed can be fully configured through a
`yaml file <https://github.com/matham/ceed/blob/master/ceed/data/CeedApp_config.yaml>`__.
This file is normally in the data directory under ceed, where ceed is installed.
Documentation for all the configuration options is listed in the
`configuration docs <https://matham.github.io/ceed/config.html>`_.
