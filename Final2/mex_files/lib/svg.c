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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "misc.h"
#include "svg.h"
#define dprint(expr) printf(#expr " = %g \n", (double) expr)

/*----------------------------------------------------------------------------*/
/*----------------------------- Write SVG Segments to File -------------------*/
/*----------------------------------------------------------------------------*/
/** Write line segments into a SVG file.
    If the name is "-" the file is written to standard output.
*/
void write_svg_segments( ntuple_list segs, char * filename,
                unsigned int xsize, unsigned int ysize, double width )
{
  FILE * svg;
  unsigned int i;

  /* open file */
  if( strcmp(filename,"-") == 0 ) svg = stdout;
  else svg = fopen(filename,"w");
  if( svg == NULL ) error("unable to open SVG output file.");

  /* write SVG header */
  fprintf(svg,"<?xml version=\"1.0\" standalone=\"no\"?>\n");
  fprintf(svg,"<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n");
  fprintf(svg," \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");
  fprintf(svg,"<svg width=\"%upx\" height=\"%upx\" ",xsize,ysize);
  fprintf(svg,"version=\"1.1\"\n xmlns=\"http://www.w3.org/2000/svg\" ");
  fprintf(svg,"xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n");

  /* write line segments */
  for(i=0;i<segs->size;i++)
    {
      fprintf(svg,"<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" ",
              segs->values[i*segs->dim+0],segs->values[i*segs->dim+1],
              segs->values[i*segs->dim+2],segs->values[i*segs->dim+3]);
      fprintf(svg,"stroke-width=\"%f\" stroke=\"black\" />\n",
              width <= 0.0 ? segs->values[i*segs->dim+4] : width);
    }

  /* close SVG file */
  fprintf(svg,"</svg>\n");
  if( svg != stdout && fclose(svg) == EOF )
    error("unable to close file while writing SVG file.");
}

/*----------------------------------------------------------------------------*/



/*----------------------------------------------------------------------------*/
/*----------------------------- Open SVG File --------------------------------*/
/*----------------------------------------------------------------------------*/
/** Opens an SVG file (writes header)
*/
FILE * open_svg_file( char * filename,
                unsigned int xsize, unsigned int ysize )
{
  FILE * svg;
  svg=fopen(filename,"w");



  /* write SVG header */
  fprintf(svg,"<?xml version=\"1.0\" standalone=\"no\"?>\n");
  fprintf(svg,"<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n");
  fprintf(svg," \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");
  fprintf(svg,"<svg width=\"%upx\" height=\"%upx\" ",xsize,ysize);
  fprintf(svg,"version=\"1.1\"\n xmlns=\"http://www.w3.org/2000/svg\" ");
  fprintf(svg,"xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n");


  fprintf(svg,"<defs>");                                 
fprintf(svg,"<linearGradient id=\"grad1\" x1=\"0%%\" y1=\"0%%\" x2=\"0%%\" y2=\"100%%\">");
fprintf(svg,"<stop offset=\"0%%\" style=\"stop-color:rgb(255,0,0);stop-opacity:1\" />");
fprintf(svg,"<stop offset=\"100%%\" style=\"stop-color:rgb(0,0,255);stop-opacity:1\" />");
fprintf(svg,"</linearGradient>");
fprintf(svg,"</defs>");                                                       

  return svg;
}

/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Close SVG FILE */
/*----------------------------------------------------------------------------*/
/** Closes an SVG file (writes footer)
*/
void close_svg_file(FILE * svg){

  /* close SVG file */
  fprintf(svg,"</svg>\n");
  if( svg != stdout && fclose(svg) == EOF )
    error("unable to close file while writing SVG file.");

}
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/* Draw segment to SVG FILE */
/*----------------------------------------------------------------------------*/

void draw_svg_segment(FILE * svg, double x1, double x2, double y1, double y2, 
double width, int r, int g, int  b, double opacity){
  /* dprint(r); */
  /* dprint(g); */
  /* dprint(b); */
  fprintf(svg,"<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" ",
	  x1,y1,x2,y2);
  fprintf(svg,"stroke-width=\"%f\" stroke=\"rgb(%d,%d,%d)\" stroke-opacity=\"%f\" />\n",
	  width/2,r,g,b,opacity);

}

void draw_svg_segment_cases(FILE * svg, double x1, double x2, double y1, double y2, 
double width, int r, int g, int  b, double opacity){
  
  fprintf(svg,"<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" ",
	  x1,y1,x2,y2);
  fprintf(svg,"stroke-width=\"%f\" stroke=\"rgb(%d,%d,%d)\" stroke-opacity=\"%f\" />\n",
	  width/2,r,g,b,opacity);

}




/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/* Draw circle to SVG FILE */
/*----------------------------------------------------------------------------*/

void draw_svg_circle(FILE * svg, double x, double y, double r){
  
  fprintf(svg,"<circle cx=\"%f\" cy=\"%f\" r=\"%f\"  ",
	  x,y,r);
  fprintf(svg,"style=\"stroke-width:1; stroke:#000000; fill:#000000\" />\n");

}



/*----------------------------------------------------------------------------*/



/*----------------------------------------------------------------------------*/
/* Draw gradient_scale to SVG FILE */
/*----------------------------------------------------------------------------*/

void draw_gradient_scale(FILE * svg, double xsize, double ysize, 
                         double vmin, double vmax){
  double width=20;
  double height=60;
  double x1=xsize-width-10;
  double y1=ysize-height-15;

  fprintf(svg,"<rect x=\"%f\" y=\"%f\" width=\"%f\" height=\"%f\" ",
          x1,y1,width,height);
  fprintf(svg," fill=\"url(#grad1)\" />\n");
    fprintf(svg, "<text fill=\"#000000\" font-size=\"10\" font-family=\"Verdana\" x=\"%f\" y=\"%f\">%.2f</text>",x1-2,y1-2,vmax);
    fprintf(svg, "<text fill=\"#000000\" font-size=\"10\" font-family=\"Verdana\" x=\"%f\" y=\"%f\">%.2f</text>",x1-2,y1+height+10,vmin);


}



/*----------------------------------------------------------------------------*/
   


