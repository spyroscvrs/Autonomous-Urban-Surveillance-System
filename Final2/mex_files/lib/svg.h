/*----------------------------------------------------------------------------

  SVG file format functions.

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
/*--------------------------------- SVG I/O ----------------------------------*/
/*----------------------------------------------------------------------------*/
/** @file svg.h
    SVG file format functions.
    @author rafael grompone von gioi (grompone@gmail.com)
 */
/*----------------------------------------------------------------------------*/
#ifndef SVG_HEADER
#define SVG_HEADER

#include "ntuple.h"

/*----------------------------------------------------------------------------*/
/*----------------------------- Write SVG File -------------------------------*/
/*----------------------------------------------------------------------------*/
/** Write line segments into a SVG file.
    If the name is "-" the file is written to standard output.
*/
void write_svg( ntuple_list segs, char * filename,
                unsigned int xsize, unsigned int ysize, double width );



/*----------------------------------------------------------------------------*/
/*----------------------------- Open SVG File --------------------------------*/
/*----------------------------------------------------------------------------*/
/** Opens an SVG file (writes header)
*/
FILE * open_svg_file( char * filename,
		    unsigned int xsize, unsigned int ysize );


/*----------------------------------------------------------------------------*/
/* Close SVG FILE */
/*----------------------------------------------------------------------------*/
/** Closes an SVG file (writes footer)
*/
void close_svg_file(FILE * svg);



/*----------------------------------------------------------------------------*/
/* Draw segment to SVG FILE */
/*----------------------------------------------------------------------------*/
void draw_svg_segment(FILE * svg, double x1, double x2, double y1, double y2, double width, int r, int g, int b, double opacity);

/*----------------------------------------------------------------------------*/
/* Draw gradient_scale to SVG FILE */
/*----------------------------------------------------------------------------*/

void draw_gradient_scale(FILE * svg, double xsize, double ysize, 
                         double vmin, double vmax);

/*----------------------------------------------------------------------------*/


#endif /* !SVG_HEADER */
/*----------------------------------------------------------------------------*/

