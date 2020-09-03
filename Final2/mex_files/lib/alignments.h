#ifndef ALIGNMENTS_HEADER
#define ALIGNMENTS_HEADER


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include "jirafa_lib.h"
#include "ntuples_aux.h"
/*----------------------------------------------------------------------------*/
struct point_alignment
{
  unsigned int i,j;    /* points used */
  double x1,y1,x2,y2;  /* points */
  double dx,dy;        /* unitary vector algon the alignment */
  double len;          /* alignment length */
  double width;        /* alignment width */
  unsigned int n;      /* number of cells */
  double s;            /* cell length */
  unsigned int k;      /* number of interior points */
  double locality;     /* locality */
  unsigned int Nlocal; /* number of points in local area */
  double po;           /* probability of having one point on a cell */
  double p;            /* probability of active cell */
  double P;            /* probability of the alignment */
  double nfa;          /* -log10(NFA) */
  ntuple_list l;     /* list of interior points */
};

/*----------------------------------------------------------------------------*/

typedef struct
{
  int pos;
  int capacity;
  struct point_alignment *array;
} alignments_list;

/*----------------------------------------------------------------------------*/
/* append alignment to alignment list */
void append_alignment( alignments_list *g, struct point_alignment *a);


/*----------------------------------------------------------------------------*/
/* append point to alignment */
void append_point_to_alignment( struct point_alignment *a, unsigned int i);
/*----------------------------------------------------------------------------*/
/** Compare alignments
 */

int compare_alignments (const void *pa, const void *pb);


/* ---------------------------------------------------------------------------*/
/** Copy an alignment structure.
 */
void copy_align( struct point_alignment * a, struct point_alignment * b );

/*----------------------------------------------------------------------------*/
/** Draw alignment from structure
 */
void draw_alignment(FILE * eps, struct point_alignment a, double ysize, 
                    double maxNFA);

/*----------------------------------------------------------------------------*/
/*                           Compare two alignments two see if one
			     masks the other  */
/* Takes two alignments A and B, where A is more significant than
   B. Answers: Is B still significant if we remove the points from
   A? */

double is_masked( ntuple_list points, struct point_alignment *a, 
		  struct point_alignment *b, double logNT, FILE * log);
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/*                           Compare two alignments WITH CASES two see if one
			     masks the other  */
/* Takes two alignments A and B, where A is more significant than
   B. Answers: Is B still significant if we remove the points from
   A? */

double is_masked_cases( ntuple_list points, struct point_alignment *a, 
		  struct point_alignment *b, double logNT);
/*----------------------------------------------------------------------------*/

/*  free alignment_list structure */
void free_alignment_list(alignments_list *a);
/*----------------------------------------------------------------------------*/



#endif /* !ALIGNMENTS_HEADER */
