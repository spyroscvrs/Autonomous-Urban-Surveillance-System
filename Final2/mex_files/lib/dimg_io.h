/*----------------------------------------------------------------------------

  DIMG image file format I/O functions.

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
/*--------------------------------- DIMG I/O ---------------------------------*/
/*----------------------------------------------------------------------------*/
/** @file dimg_io.h
    DIMG format read and write functions.
    @author rafael grompone von gioi (grompone@gmail.com)

    DIMG, ASCII double image file format:

    DoubleImageAscii <xsize> <ysize> <val0> <val1> <val2>...

    Separator between tokens can be any "space" character,
    as space, tabs, end-of-line. Values should be written
    in %g printf format. Values appear line by line,
    so the first value has coordinates x=0,y=0, the second
    one is x=1,y0, until x=<maxX>,y=0; the follwing one
    is x=0,y=1, and so on.

    For example, the following is a 5x5 image:
    DoubleImageAscii
    5 5
    0 0.1 10.2 100 2
    -1 -1 -1 -1 -1
    -1 -2 -3 -4 -5
    1 2 3 4 5
    1 1 1 1 1
 */
/*----------------------------------------------------------------------------*/
#ifndef DIMG_IO_HEADER
#define DIMG_IO_HEADER

#include "image.h"

/*----------------------------------------------------------------------------*/
/** Read a DIMG file into an "image_double".
    If the name is "-" the file is read from standard input.
 */
image_double read_dimg_image_double(char * name);

/*----------------------------------------------------------------------------*/
/** Write an "image_double" into a DIMG file.
    If the name is "-" the file is written to standard output.
 */
void write_dimg_image_double(image_double image, char * name);

#endif /* !DIMG_IO_HEADER */
/*----------------------------------------------------------------------------*/
