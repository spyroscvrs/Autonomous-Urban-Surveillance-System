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
/** @file eps.c
    EPS PostScript file format functions.
    @author rafael grompone von gioi (grompone@gmail.com)
 */
/*----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "misc.h"
#include "eps.h"

/*----------------------------------------------------------------------------*/
/** Open an EPS file.
    If the name is "-" the file is written to standard output.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
FILE * eps_open(char * filename, unsigned int xsize, unsigned int ysize)
{
  FILE * eps;

  /* open file */
  if( strcmp(filename,"-") == 0 ) eps = stdout;
  else eps = fopen(filename,"w");
  if( eps == NULL ) error("unable to open EPS output file.");

  /* write EPS header */
  fprintf(eps,"%%!PS-Adobe-3.0 EPSF-3.0\n");
  fprintf(eps,"%%%%BoundingBox: 0 0 %u %u\n",xsize,ysize);
  fprintf(eps,"%%%%Creator: jirafa_lib\n");
  fprintf(eps,"%%%%Title: (%s)\n",filename);
  fprintf(eps,"%%%%EndComments\n");

  return eps;
}

/*----------------------------------------------------------------------------*/
/** Close an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_close(FILE * eps)
{
  if( eps == NULL ) error("NULL EPS file pointer.");

  /* close EPS file */
  fprintf(eps,"showpage\n");
  fprintf(eps,"%%%%EOF\n");
  if( eps != stdout && fclose(eps) == EOF )
    error("unable to close file while writing EPS file.");
}

/*----------------------------------------------------------------------------*/
/** Add a line segment to an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_add_line( FILE * eps, double x1, double y1, double x2, double y2,
                   double width )
{
  if( eps == NULL ) error("NULL EPS file pointer.");

  fprintf( eps, "newpath %f %f moveto %f %f lineto %f setlinewidth stroke\n",
           x1, y1, x2, y2, width );
}

/*----------------------------------------------------------------------------*/
/** Add a dot to an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_add_dot( FILE * eps, double x, double y, double radius )
{
  if( eps == NULL ) error("NULL EPS file pointer.");
  fprintf(eps,"newpath %f %f %f 0 360 arc closepath fill\n",x,y,radius);
}

/*----------------------------------------------------------------------------*/
/** Add a circle to an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_add_circle( FILE * eps, double x, double y, double radius )
{
  if( eps == NULL ) error("NULL EPS file pointer.");
  fprintf(eps,"newpath %f %f %f 0 360 arc closepath stroke\n",x,y,radius);
}

/*----------------------------------------------------------------------------*/
/** Set current color of an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_set_current_color( FILE * eps, double r, double g, double b )
{
  if( eps == NULL ) error("eps_set_current_color: NULL EPS file pointer.");
  if( r < 0.0 || r > 1.0 || g < 0.0 || g > 1.0 || b < 0.0 || b > 1.0 )
    error("eps_set_current_color: r, g, and b must take values in [0,1]");
  fprintf(eps,"%f %f %f setrgbcolor\n",r,g,b);
}

/*----------------------------------------------------------------------------*/
/** Write a text to an EPS file.

    According to

      Adobe "Encapsulated PostScript File Format Specification",
      Version 3.0, 1 May 1992,

    and

      Adobe "PostScript(R) LANGUAGE REFERENCE", third edition, 1999.
 */
void eps_write( FILE * eps, double fontsize, double x, double y, char * txt )
{
  if( eps == NULL ) error("eps_set_current_color: NULL EPS file pointer.");
  fprintf(eps,"/Helvetica-Bold findfont %f scalefont setfont\n",fontsize);
  fprintf(eps,"%f %f moveto (%s) show\n",x,y,txt);
}

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
                      unsigned int xsize, unsigned int ysize, double width )
{
  FILE * eps;
  unsigned int i;

  /* open file */
  eps = eps_open(filename, xsize, ysize);

  /* write line segments */
  for(i=0;i<segs->size;i++)
    eps_add_line( eps, segs->values[i*segs->dim+0],
                  (double) ysize - segs->values[i*segs->dim+1],
                  segs->values[i*segs->dim+2],
                  (double) ysize - segs->values[i*segs->dim+3],
                  width <= 0.0 ? segs->values[i*segs->dim+4] : width );

  /* close EPS file */
  eps_close(eps);
}

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Write rectangles into an EPS file.
    If the name is "-" the file is written to standard output.

    According to
    
    Rafael Grompone's code :)
 */

/* void eps_write_rectangle(FILE * eps, double x1, double x2, double y1, double y2){ */
/*   fprintf(eps,"newpath %f %f moveto %f %f lineto ", */
/*           best_align.x1 - best_align.dy * best_align.width * 0.5, */
/*           ysize - best_align.y1 - best_align.dx * best_align.width * 0.5, */
/*           best_align.x2 - best_align.dy * best_align.width * 0.5, */
/*           ysize - best_align.y2 - best_align.dx * best_align.width * 0.5); */



