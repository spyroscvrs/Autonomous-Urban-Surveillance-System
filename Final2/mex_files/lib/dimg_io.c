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
/*----------------------------- double_image I/O -----------------------------*/
/*----------------------------------------------------------------------------*/
/** @file dimg_io.c
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "misc.h"
#include "image.h"
#include "dimg_io.h"

/*----------------------------------------------------------------------------*/
/** DIMG file identificator
 */
#define DIMG_ID "DoubleImageAscii"

/*----------------------------------------------------------------------------*/
/** Read a DIMG file into an "image_double".
    If the name is "-" the file is read from standard input.
 */
image_double read_dimg_image_double(char * name)
{
  FILE * f;
  int i,n;
  unsigned int xsize,ysize,x,y;
  double val;
  image_double image;
  char * file_id = DIMG_ID;

  /* open file */
  if( strcmp(name,"-") == 0 ) f = stdin;
  else f = fopen(name,"r");
  if( f == NULL ) error("unable to open input image file.");

  /* read header */
  n = strlen(file_id);
  for(i=0;i<n;i++)
    if( getc(f) != file_id[i] ) error("not a DIMG file!");
  n = fscanf(f,"%u",&xsize);
  if(n<1) error("not a DIMG file!");
  n = fscanf(f,"%u",&ysize);
  if(n<1) error("not a DIMG file!");

  /* get memory */
  image = new_image_double(xsize,ysize);

  /* read data */
  for(y=0;y<ysize;y++)
    for(x=0;x<xsize;x++)
      {
        n = fscanf(f,"%lg",&val);
        if(n<1) error("corrupted DIMG file!");
        image->data[ x + y * xsize ] = val;
      }

  /* close file if needed */
  if( f != stdin && fclose(f) == EOF )
    error("unable to close file while reading DIMG file.");

  return image;
}

/*----------------------------------------------------------------------------*/
/** Write an "image_double" into a DIMG file.
    If the name is "-" the file is written to standard output.
 */
void write_dimg_image_double(image_double image, char * name)
{
  FILE * f;
  unsigned int x,y,n;

  /* open file */
  if( strcmp(name,"-") == 0 ) f = stdout;
  else f = fopen(name,"w");
  if( f == NULL ) error("unable to open output image file.");

  /* write header */
  fprintf(f,"%s\n",DIMG_ID);
  fprintf(f,"%u %u\n",image->xsize,image->ysize);

  /* write data */
  for(n=0,y=0; y<image->ysize; y++)
    for(x=0; x<image->xsize; x++)
      {
        fprintf(f,"%g ",image->data[ x + y * image->xsize ]);
        if(++n==8)  /* to keep lines short, just to be nice :) */
          {
            fprintf(f,"\n");
            n = 0;
          }
      }

  /* close file if needed */
  if( f != stdout && fclose(f) == EOF )
    error("unable to close file while writing DIMG file.");
}

/*----------------------------------------------------------------------------*/
