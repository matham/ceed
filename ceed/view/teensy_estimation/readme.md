Teensy Frame Estimation
=======================

The Teensy is used to help Ceed identify when a frame was rendered for two frames by the GPU, requiring
Ceed to drop a frame. See more details in the [docs](https://matham.github.io/ceed/blueprint.html).

Programming Teensy
------------------

If this is a new Teensy device, it'll need to be re-programmed. The easiest way is to download the
pre-compiled
[teensy_estimation.ino.hex file](./teensy_estimation.ino.hex).
Then, following the instructions and using the [Teensy Loader App](https://www.pjrc.com/teensy/loader.html),
program the Teensy with the hex file.

If something has changed and the hex file needs to be recompiled, then follow the
[instructions](https://www.pjrc.com/teensy/teensyduino.html) to install Teensyduino by installing the
[Arduino IDE](https://www.arduino.cc/en/software) and then the
[Teensyduino add-on](https://www.pjrc.com/teensy/td_download.html).

Then, get and open the source
[teensy_estimation.ino file](./teensy_estimation.ino).
Then, locate the ``teensy4/usb_rawhid.c`` file, it'll be located under the arduino installation
(e.g. ``arduino-1.8.13/hardware/teensy/avr/cores/teensy4/usb_rawhid.c``). You'll have to make three changes in the file
to improve latency:

* change ``#define TX_NUM   4`` to ``#define TX_NUM   1``,
* in the function ``usb_rawhid_recv`` change the line ``if (systick_millis_count - wait_begin_at > timeout)  {`` to
  ``if (systick_millis_count - wait_begin_at >= timeout)  {``
* similarly, in the function ``usb_rawhid_send`` change the line ``if (systick_millis_count - wait_begin_at > timeout) return 0;``
  to ``if (systick_millis_count - wait_begin_at >= timeout) return 0;``

Then after saving, in the Arduino IDE ensure that the USB type under tools is set to "Raw HID". And that for the
board the Teensy 4.1 is selected. And finally, under port, select the Teensy device so it can be programmed.
Finally, compile ``teensy_estimation.ino`` and upload it to the Teensy as described in the
[docs](https://www.pjrc.com/teensy/td_usage.html).
