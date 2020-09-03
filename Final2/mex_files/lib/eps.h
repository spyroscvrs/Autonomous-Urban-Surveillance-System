/*----------------------------------------------------------------------------

  EPS file generation functions.

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
/*--------------------------------- EPS I/O ----------------------------------*/
/*----------------------------------------------------------------------------*/
/** @file eps.h
    EPS PostScript file format functions.
    @author rafael grompone von gioi (grompone@gmail.com)
 */
/*----------------------------------------------------------------------------*/
#ifndef EPS_HEADER
#define EPS_HEADER

#include <stdio.h>
#include <stdlib.h>
#include "ntuple.h"

/*----------------------------------------------------------------------------*/
/** Open an EPS file.
    If the name is "-" the file is written to standard output.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
FILE * eps_open(char * filename, unsigned int xsize, unsigned int ysize);

/*----------------------------------------------------------------------------*/
/** Close an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_close(FILE * eps);

/*----------------------------------------------------------------------------*/
/** Add a line segment to an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_add_line( FILE * eps, double x1, double y1, double x2, double y2,
                   double width );

/*----------------------------------------------------------------------------*/
/** Add a dot to an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_add_dot( FILE * eps, double x, double y, double radius );

/*----------------------------------------------------------------------------*/
/** Add a circle to an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_add_circle( FILE * eps, double x, double y, double radius );

/*----------------------------------------------------------------------------*/
/** Set current color of an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_set_current_color( FILE * eps, double r, double g, double b );

/*----------------------------------------------------------------------------*/
/** Write a text to an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_write( FILE * eps, double fontsize, double x, double y, char * txt );

/*----------------------------------------------------------------------------*/
/** Write line segments into an EPS file.
    If the name is "-" the file is written to standard output.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_write_lines( ntuple_list segs, char * filename,
                      unsigned int xsize, unsigned int ysize, double width );

#endif /* !EPS_HEADER */
/*----------------------------------------------------------------------------*/
