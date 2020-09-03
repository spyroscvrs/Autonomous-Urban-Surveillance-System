/*----------------------------------------------------------------------------

  Simple X GUI functions.

  Copyright 2010-2011 rafael grompone von gioi (grompone@gmail.com)

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  ----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*------------------------------- Simple X GUI -------------------------------*/
/*----------------------------------------------------------------------------*/
/** @file x_gui.h
    Simple Graphical User Interface based on Xlib.
    @author rafael grompone von gioi (grompone@gmail.com)

    It is only active it USE_X_GUI is defined, and it
    needs the library Xlib.
 */
/*----------------------------------------------------------------------------*/
#ifndef XGUI_HEADER
#define XGUI_HEADER
#include "image.h"

/*----------------------------------------------------------------------------*/
/** Initialize GUI.
 */
void x_gui_init(unsigned int xsize, unsigned int ysize);

/*----------------------------------------------------------------------------*/
/** Close GUI.
 */
void x_gui_close(void);

/*----------------------------------------------------------------------------*/
/** Display an image_char to the window.
 */
void x_gui_display_image_char(image_char in);

/*----------------------------------------------------------------------------*/
/** Display an image_int to the window.
 */
void x_gui_display_image_int(image_int in);

/*----------------------------------------------------------------------------*/
/** Display an image_double to the window.
 */
void x_gui_display_image_double(image_double in);

/*----------------------------------------------------------------------------*/
/** Wait for a click on the window and returns its coordinates
    and button number (1 to 5).
 */
int x_gui_get_click(int * x, int * y);

/*----------------------------------------------------------------------------*/
/** Wait for a click on the window and returns button number (1 to 5).
 */
int x_gui_wait_click(void);

/*----------------------------------------------------------------------------*/
/** Wait until a keyboard key is pressed; return its xlib code.
 */
unsigned int x_gui_get_key(void);

/*----------------------------------------------------------------------------*/
/** Draw a dot on the window centered at x,y.
 */
void x_gui_draw_point(int x, int y);

/*----------------------------------------------------------------------------*/
/** Draw a dot on the window centered at x,y and radius rad.
 */
void x_gui_draw_point_rad(int x, int y, unsigned int rad);

/*----------------------------------------------------------------------------*/
/** Draw a line from x1,y1 to x2,y2.
 */
void x_gui_draw_line(int x1, int y1, int x2, int y2);

/*----------------------------------------------------------------------------*/
/** Write text 'str' at position x,y.
 */
void x_gui_write(int x, int y, char * str);

/*----------------------------------------------------------------------------*/
/** Set foreground color.

    ex. "red".
 */
void x_gui_foreground_color(char * color_name);

#endif /* !XGUI_HEADER */
/*----------------------------------------------------------------------------*/
