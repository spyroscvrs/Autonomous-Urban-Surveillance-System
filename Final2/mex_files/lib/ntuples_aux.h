#ifndef NTUPLE_AUX_HEADER
#define NTUPLE_AUX_HEADER

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include "jirafa_lib.h"

/*----------------------------------------------------------------------------*/
/* append value to ntuple list */
void append_to_ntuple( ntuple_list a, double x);

/*----------------------------------------------------------------------------*/
/* copy ntuple */

ntuple_list copy_ntuple_list(ntuple_list  a);

/* ---------------------------------------------------------------------------*/
/* Check if a 1D-ntuple contains an item */
int is_in_ntuple(ntuple_list l, double x);

/*----------------------------------------------------------------------------*/
/** Read TXT file with point coordinates.
 */
ntuple_list read_points(char * filename);

/* ---------------------------------------------------------------------------*/

/* ---------------------------------------------------------------------------*/
#endif /* !NTUPLE_AUX_HEADER */
