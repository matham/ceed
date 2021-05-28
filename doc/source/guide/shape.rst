Ceed shapes
===========

A Ceed :ref:`shape <shape-api>` is how a stage sets the color intensity of a region
of space. For every frame each stage gets an intensity value from its function and it
sets all the shapes and shape groups attached to that stage to that intensity.

The main black area of Ceed is a drawing area the size of the projector, in pixels,
upon which shapes are drawn.

Drawing shape
-------------

To add a shape, first ensure you're in ``draw`` mode, then ensure that one of the 4
drawing types is selected. You can draw a circle, ellipse, freeform polygon, and a
polygon.

Drop one or more points depending on the shape type to create the shape. Once created, a
long-press on a point will activate edit mode for that shape and you can interact with
any of the points of that shape. Clicking on black-space will exit editing that shape.

If no shape is being edited, dragging the orange point will drag the shape. If none of the
4 shape drawing types is selected, you can only edit the shapes, but not create new ones.

.. video:: ../media/guide/shape_draw.webm

Displaying shapes
-----------------

Once created, shapes can be controlled in the shapes pane. Using the buttons a shape can
be locked from interaction, or it can be hidden altogether. It can also be deleted.

There's an implicit shape depth order. Shapes further down in the list are above shapes
earlier in the list. If two shapes occupy the same space or overlap, the shape on top
is used for the region where they overlap. The arrows allow moving a shape up or down.
See :ref:`donut stage <stage-donut>` how to use depth to create a donut shaped stage.

Selecting a shape in the pane will also highlight it in the drawing area. There, you
can change its name to something unique. You can change the x, y position of its
centroid. And you can set its approximate area and it'll be automatically scaled
to the closest feasible area.

.. video:: ../media/guide/shape_control.webm

.. _control-shape:

Controlling shapes
------------------

By default, you can only select one shape at a time. Enabling multiselect in the
controls, or holding down the ctrl key enabled selecting multiples shape, either in
the pane or by touching their orange point.

Dragging on the orange point of one will drag all the selected shapes.
From the controls you can also duplicate the selected shapes, or delete them.

There's a special button that adds a polygon shape named "enclosed" that encloses the
whole projector area. It is typically used to stimulate the entire slice.

From the controls you can also enable a mode to see the name of each shape and the x, y
pixel position of the mouse as it transverses the drawing area.

.. video:: ../media/guide/shape_multiple.webm

Shape groups
------------

Rather than dragging individual shapes into stages, you can group shapes into shape groups.
Then, these groups can be added to stages and it's as if all it shapes are added to the stage.
A single shape may be in multiple groups and all these groups may be added to a stage
simultaneously, as well as the individual shape itself, without problems.

.. video:: ../media/guide/shape_groups.webm
