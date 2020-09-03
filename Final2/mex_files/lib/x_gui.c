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
/** @file x_gui.c
    Simple Graphical User Interface based on Xlib.
    @author rafael grompone von gioi (grompone@gmail.com)

    It is only active it USE_X_GUI is defined, and it
    needs the library Xlib.
 */
/*----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include "misc.h"
#include "image.h"

/*----------------------------------------------------------------------------*/
#ifdef USE_X_GUI
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif /* USE_X_GUI */

/*----------------------------------------------------------------------------*/
/* static variables to hold elements references */
#ifdef USE_X_GUI
static Display * display = NULL;
static Window window;
static XImage * image = NULL;
static int screen, depth;
static uint32_t * buffer = NULL;
#endif /* USE_X_GUI */

/*----------------------------------------------------------------------------*/
/** Determine if the systems is little-endian or big-endian.
    www.xmission.com/~georgeps/documentation/tutorials/Xlib_JPEG_Tutorial-6.html
 */
#ifdef USE_X_GUI
static int get_byte_order(void)
{
  union
  {
    char c[sizeof(short)];
    short s;
  } order;

  order.s = 1;

  if( order.c[0] == 1 ) return LSBFirst;
  else return MSBFirst;
}
#endif /* USE_X_GUI */

/*----------------------------------------------------------------------------*/
/** Initialize GUI.
 */
void x_gui_init(unsigned int xsize, unsigned int ysize)
{
#ifdef USE_X_GUI
  /* open X conection */
  display = XOpenDisplay(NULL);
  if( display == NULL ) error("can't open X server.");
  screen = DefaultScreen(display);

  /* create XImage */
  depth = DefaultDepth(display,screen);
  if(depth<24) error("pixel depth of less than 24 bits are not supported.");

  buffer = (uint32_t *) malloc( (size_t) (4 * xsize * ysize) );
  if( buffer == NULL ) error("not enough memory.");

  image = XCreateImage( display, CopyFromParent, depth, ZPixmap, 0, 
                        (char *) buffer, xsize, ysize, 32, 0 );
  XInitImage(image);
  if( get_byte_order() == LSBFirst ) image->byte_order = LSBFirst;
  else image->byte_order = MSBFirst;
  image->bitmap_bit_order = MSBFirst;

  /* create window */
  window = XCreateWindow( display, DefaultRootWindow(display), 0, 0,
                          xsize, ysize, 0, CopyFromParent,
                          CopyFromParent, CopyFromParent, 0, NULL );
  XMapWindow(display, window);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Close GUI.
 */
void x_gui_close(void)
{
#ifdef USE_X_GUI
  XDestroyImage(image);
  XDestroyWindow(display,window);
  XCloseDisplay(display);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Display an image_char to the window.
 */
void x_gui_display_image_char(image_char in)
{
#ifdef USE_X_GUI
  static Visual * visual;
  double r_ratio,g_ratio,b_ratio;
  unsigned int r,g,b,x,y;
  char v;

  /* check if the image fit */
  if( (int) in->xsize > image->width || (int) in->ysize > image->height )
    error("x_gui_display_image_char: image to show do not fit on window.");

  visual = DefaultVisual(display,screen);
  r_ratio = visual->red_mask   / 255.0;
  g_ratio = visual->green_mask / 255.0;
  b_ratio = visual->blue_mask  / 255.0;

  /* copy image to buffer */
  for(x=0; x<in->xsize; x++)
    for(y=0; y<in->ysize; y++)
      {
        v = in->data[ x + y * in->xsize ];
        r = v * r_ratio;
        g = v * g_ratio;
        b = v * b_ratio;
        r &= visual->red_mask;
        g &= visual->green_mask;
        b &= visual->blue_mask;
        buffer[ x + y * image->width ] = r | g | b;
      }

  /* put image on window */
  XPutImage( display, window, DefaultGC(display,screen),
             image, 0, 0, 0, 0, in->xsize, in->ysize );
  XFlush(display);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Display an image_int to the window.
 */
void x_gui_display_image_int(image_int in)
{
#ifdef USE_X_GUI
  static Visual * visual;
  double r_ratio,g_ratio,b_ratio;
  unsigned int r,g,b,x,y;
  int max,min,v;

  /* check if the image fit */
  if( (int) in->xsize > image->width || (int) in->ysize > image->height )
    error("x_gui_display_image_int: image to show do not fit on window.");

  max = INT_MIN;
  min = INT_MAX;
  for(x=0; x<in->xsize; x++)
    for(y=0; y<in->ysize; y++)
      {
        v = in->data[ x + y * in->xsize ];
        if( v > max ) max = v;
        if( v < min ) min = v;
      }

  visual = DefaultVisual(display,screen);
  r_ratio = visual->red_mask   / (double) (max-min);
  g_ratio = visual->green_mask / (double) (max-min);
  b_ratio = visual->blue_mask  / (double) (max-min);

  /* copy image to buffer */
  for(x=0; x<in->xsize; x++)
    for(y=0; y<in->ysize; y++)
      {
        v = in->data[ x + y * in->xsize ] - min;
        r = v * r_ratio;
        g = v * g_ratio;
        b = v * b_ratio;
        r &= visual->red_mask;
        g &= visual->green_mask;
        b &= visual->blue_mask;
        buffer[ x + y * image->width ] = r | g | b;
      }

  /* put image on window */
  XPutImage( display, window, DefaultGC(display,screen),
             image, 0, 0, 0, 0, in->xsize, in->ysize );
  XFlush(display);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Display an image_double to the window.
 */
void x_gui_display_image_double(image_double in)
{
#ifdef USE_X_GUI
  static Visual * visual;
  double r_ratio,g_ratio,b_ratio;
  unsigned int r,g,b,x,y;
  double max,min,v;

  /* check if the image fit */
  if( (int) in->xsize > image->width || (int) in->ysize > image->height )
    error("x_gui_display_image_double: image to show do not fit on window.");

  max = -DBL_MIN;
  min = DBL_MAX;
  for(x=0; x<in->xsize; x++)
    for(y=0; y<in->ysize; y++)
      {
        v = in->data[ x + y * in->xsize ];
        if( v > max ) max = v;
        if( v < min ) min = v;
      }

  visual = DefaultVisual(display,screen);
  r_ratio = visual->red_mask   / (max-min);
  g_ratio = visual->green_mask / (max-min);
  b_ratio = visual->blue_mask  / (max-min);

  /* copy image to buffer */
  for(x=0; x<in->xsize; x++)
    for(y=0; y<in->ysize; y++)
      {
        v = in->data[ x + y * in->xsize ] - min;
        r = v * r_ratio;
        g = v * g_ratio;
        b = v * b_ratio;
        r &= visual->red_mask;
        g &= visual->green_mask;
        b &= visual->blue_mask;
        buffer[ x + y * image->width ] = r | g | b;
      }

  /* put image on window */
  XPutImage( display, window, DefaultGC(display,screen),
             image, 0, 0, 0, 0, in->xsize, in->ysize );
  XFlush(display);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Wait for a click on the window and returns its coordinates
    and button number (1 to 5).
 */
int x_gui_get_click(int * x, int * y)
{
#ifdef USE_X_GUI
  XEvent event;

  XSelectInput( display, window, ButtonPressMask );
  XNextEvent(display, &event);

  *x = event.xbutton.x - 1; /* to give coordinates starting at 0,0 */
  *y = event.xbutton.y - 1;

  if( event.xbutton.button == Button1 ) return 1;
  else if( event.xbutton.button == Button2 ) return 2;
  else if( event.xbutton.button == Button3 ) return 3;
  else if( event.xbutton.button == Button4 ) return 4;
  else return 5;
#else
  return 0;
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Wait for a click on the window and returns button number (1 to 5).
 */
int x_gui_wait_click(void)
{
#ifdef USE_X_GUI
  int x,y;
  return x_gui_get_click(&x,&y);
#else
  return 0;
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Wait until a keyboard key is pressed; return its xlib code.

    UNFINISHED!!!
 */
unsigned int x_gui_get_key(void)
{
#ifdef USE_X_GUI
  XEvent event;

  XSelectInput( display, window, KeyPressMask );
  XNextEvent(display, &event);

  char str[25];
  int len;
  KeySym keysym;
  len = XLookupString(&event.xkey,str,25,&keysym,NULL);
  printf("%s %d\n",str,str[0]);

  return event.xkey.keycode;
#else
  return 0;
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Draw a dot on the window centered at x,y.
 */
void x_gui_draw_point(int x, int y)
{
#ifdef USE_X_GUI
  /*
  XColor color;
  XParseColor(display,DefaultColormap(display,screen),"red", &color);
  XAllocColor(display,DefaultColormap(display,screen),&color);
  XSetForeground(display,DefaultGC(display,screen),color.pixel);
  */

  /*
  XDrawPoint(display, window, DefaultGC(display,screen), x, y);
  XDrawArc(display,window,DefaultGC(display,screen),x-3,y-3,7,7,0,23040);
  */
  XFillArc(display,window,DefaultGC(display,screen),x-3,y-3,7,7,0,23040);
  XFlush(display);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Draw a dot on the window centered at x,y and radius rad.
 */
void x_gui_draw_point_rad(int x, int y, unsigned int rad)
{
#ifdef USE_X_GUI
  XFillArc( display, window, DefaultGC(display,screen),
            x-rad, y-rad, 2*rad+1, 2*rad+1, 0, 23040 );
  XFlush(display);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Draw a line from x1,y1 to x2,y2.
 */
void x_gui_draw_line(int x1, int y1, int x2, int y2)
{
#ifdef USE_X_GUI
  XDrawLine(display,window,DefaultGC(display,screen),x1,y1,x2,y2);
  XFlush(display);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Write text 'str' at position x,y.
 */
void x_gui_write(int x, int y, char * str)
{
#ifdef USE_X_GUI
  /*
  XColor color;
  XParseColor(display,DefaultColormap(display,screen),"red", &color);
  XAllocColor(display,DefaultColormap(display,screen),&color);
  XSetForeground(display,DefaultGC(display,screen),color.pixel);
  */

  XDrawString(display,window,DefaultGC(display,screen),x,y,str,strlen(str));
  XFlush(display);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
/** Set foreground color.

    ex. "red".
 */
void x_gui_foreground_color(char * color_name)
{
#ifdef USE_X_GUI
  XColor color;
  XParseColor(display,DefaultColormap(display,screen),color_name, &color);
  XAllocColor(display,DefaultColormap(display,screen),&color);
  XSetForeground(display,DefaultGC(display,screen),color.pixel);
#endif /* USE_X_GUI */
}

/*----------------------------------------------------------------------------*/
