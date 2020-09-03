#include "alignments.h"

#define dprint(expr) printf(#expr " = %g \n", (double) expr)
#define PI 3.14159265358979
/*----------------------------------------------------------------------------*/
/* append alignment to alignment list */
void append_alignment( alignments_list *g, struct point_alignment *a){
  if (g->pos == g->capacity) { /* if array is full, duplicate size */
    g->capacity++;
    g->array = (struct point_alignment*)realloc(g->array, 
                                       ((g->capacity) *= 2) 
                                                *(sizeof(struct point_alignment)
                                                  +sizeof(ntuple_list)));
  }

  struct point_alignment b;
  
  copy_align(a,&b);

  g->array[g->pos++] = b; /* append to array */

  /* printf("inside function: dimension of a is %d\n",a->l->dim); */
}


/*----------------------------------------------------------------------------*/
/* append point to alignment */
void append_point_to_alignment( struct point_alignment *a, unsigned int i){
  if (a->l->size == a->l->max_size) enlarge_ntuple_list(a->l);
  a->l->values[(a->l->size)*(a->l->dim)]=(double) i;
  a->l->size++;

}
/*----------------------------------------------------------------------------*/
/** Compare alignments
 */

int compare_alignments (const void *pa, const void *pb)
{

  struct point_alignment *a = (struct point_alignment*)pa;
  struct point_alignment *b = (struct point_alignment*)pb;


  double da = a->nfa;
  double db = b->nfa;
  if(da!=db)
    return (da > db) - (da < db);
  else /* if NFA is equal give preference to the longest one */
    return (a->len > b->len) - (a->len < b->len);
    /* return (a->width < b->width) - (a->width > b->width); */

}


/* ---------------------------------------------------------------------------*/
/** Copy an alignment structure.
 */
void copy_align( struct point_alignment * a, struct point_alignment * b )
{
  if( a == NULL || b == NULL ) error("copy_align: NULL input pointers.");

  b->i = a->i;
  b->j = a->j;
  b->x1 = a->x1;
  b->y1 = a->y1;
  b->x2 = a->x2;
  b->y2 = a->y2;
  b->dx = a->dx;
  b->dy = a->dy;
  b->len = a->len;
  b->width = a->width;
  b->n = a->n;
  b->s = a->s;
  b->k = a->k;
  b->locality = a->locality;
  b->Nlocal = a->Nlocal;
  b->po = a->po;
  b->p = a->p;
  b->P = a->P;
  b->nfa = a->nfa;

  /* list of interior points */
  b->l= copy_ntuple_list(a->l);

}


/*----------------------------------------------------------------------------*/
/** Draw alignment from structure
 */
void draw_alignment(FILE * eps, struct point_alignment a, double ysize, 
                    double maxNFA){
  double red = a.P/maxNFA; /* 1 if most significant, 0 if not */
  
  fprintf(eps,"%g 0 %g setrgbcolor\n",red,1-red); /* set color to back */
  /* draw alignemnt rectangle */
  fprintf(eps,"newpath %f %f moveto %f %f lineto ",
          a.x1 - a.dy * a.width * 0.5,
          ysize - a.y1 - a.dx * a.width * 0.5,
          a.x2 - a.dy * a.width * 0.5,
          ysize - a.y2 - a.dx * a.width * 0.5);
  fprintf(eps,"%f %f lineto %f %f lineto closepath ",
          a.x2 + a.dy * a.width * 0.5,
          ysize - a.y2 + a.dx * a.width * 0.5,
          a.x1 + a.dy * a.width * 0.5,
          ysize - a.y1 + a.dx * a.width * 0.5);
  fprintf(eps,"gsave fill grestore\n");
  fprintf(eps,"%f setlinewidth stroke\n",0.5);
}




/*----------------------------------------------------------------------------*/
/*                           Compare two alignments two see if one
			     masks the other  */
/* Takes two alignments A and B, where A is more significant than
   B. Answers: Is B still significant if we remove the points from
   A? */

double is_masked( ntuple_list points, struct point_alignment *a, 
		  struct point_alignment *b, double logNT, FILE * log){
  int i;
  double NFA;
  int diff;
  int k2 = b->k; 		/* number of points in b */


  /* Check the anlge between rectangles before proceeding. This operation should
     be avoided in the case without boxes because every cluster with another
     point forms an alignment. Evaluating only the angle will result in many
     unwanted detecions */

  /* double costheta, theta; */

  /* costheta = (a->x2 - a->x1) * (b->x2 - b->x1) + (a->y2 - a->y1) * (b->y2 - b->y1); */
  
  /* costheta/= sqrt(pow(a->x2 - a->x1,2) + pow(a->y2 - a->y1,2)); */
  /* costheta/= sqrt(pow(b->x2 - b->x1,2) + pow(b->y2 - b->y1,2)); */

  /* theta = acos(costheta); */
  /* theta = fmin(theta, PI-theta); */

  /* if( theta > PI/6.) */
  /*   return 10.0; */


 

 

  for (i=0;i<b->l->size;i++){
    if (is_in_ntuple(a->l,b->l->values[i])) k2--; /* decrease the
                                                     number of points
						     in b */
  }
  /* recompute b's NFA */
  diff = b->k - k2;
  if (k2>0) {
    /* dprint(k2); */
    NFA = nfa(b->Nlocal-diff,k2,b->po,logNT);
  }
  else NFA= -DBL_MAX;
    
  return NFA;

}
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/*                           Compare two alignments  WITH CASES to see if one
			     masks the other  */
/* Takes two alignments A and B, where A is more significant than
   B. Answers: Is B still significant if we remove the points from
   A? */

double is_masked_cases( ntuple_list points, struct point_alignment *a, 
		  struct point_alignment *b, double logNT){
  int i;
  double NFA;
  int diff;
  int k2 = b->k; 		/* number of points in b */



  /* Check the anlge between rectangles before proceeding. This operation should
     be avoided in the case without boxes because every cluster with another
     point forms an alignment. Evaluating only the angle will result in many
     unwanted detecions */

   /* vec1 = [x2a-x1a,y2a-y1a]; */
   /* vec2 = [x2b-x1b,y2b-y1b]; */
   /* cos(theta)=vec1(1)*vec2(1)+vec1(2)*vec2(2)/norm(vec1)/norm(vec2);? */
   /* theta = acos(cos(theta); */


  double costheta, theta;

  costheta = (a->x2 - a->x1) * (b->x2 - b->x1) + (a->y2 - a->y1) * (b->y2 - b->y1);
  
  costheta/= sqrt(pow(a->x2 - a->x1,2) + pow(a->y2 - a->y1,2));
  costheta/= sqrt(pow(b->x2 - b->x1,2) + pow(b->y2 - b->y1,2));

  theta = acos(costheta);
  theta = fmin(theta, PI-theta);

  if( theta > PI/10.)
    return 10.0;
               
    


  


  struct point_alignment bnew;  /* create a new alignment with points that are
                                   not in a */

  bnew.l=new_ntuple_list(1);

  bnew.i=b->i;
  bnew.j=b->j;
  bnew.x1=b->x1;
  bnew.y1=b->y1;
  bnew.x2=b->x2;
  bnew.y2=b->y2;
  bnew.len=b->len;
  
  bnew.dx=b->dx;
  bnew.dy=b->dy;
  
  bnew.width=b->width;
  int n;
  n=bnew.n=b->n;
  bnew.s=b->s;
  bnew.po = b->po;

  bnew.Nlocal = b->Nlocal;      /* WARNING: Nlocal could include points in a and
                                   should be reduced */

  int * cells;
  cells = (int *) calloc( (size_t) (n+1), sizeof(int) );
  if( cells == NULL ) error("not enough memory.");

  bnew.k = 0;
  
  int l;
  double x,y, fwd_dist;


  for(l=0;l<n;l++) {
    cells[l] = 0;
  }

  /* if point in b is not in a, add to bnew  */
  for (i=0;i<b->l->size;i++){
    if (!is_in_ntuple(a->l,b->l->values[i])){
      int l=b->l->values[i];
      x = points->values[ l*points->dim + 0 ];
      y = points->values[ l*points->dim + 1 ];
      fwd_dist = (x-bnew.x1)*bnew.dx + (y-bnew.y1)*bnew.dy;
      cells[ (int) (fwd_dist/bnew.s - 0.5) ] = 1;
    } 
  }

  for(l=0;l<n;l++) bnew.k += cells[l];

  /* printf("bnew:\n"); */
  /* dprint(bnew.k); */
  /* dprint(b->k); */

  /* compute event probability */
  
  bnew.p = 1.0 - pow( 1.0 - bnew.po, (double) bnew.Nlocal );
  if( bnew.p > 0.0 && bnew.p < 1.0 ){
    bnew.nfa = nfa(n,bnew.k,bnew.p,logNT);
  }
  else bnew.nfa =  -DBL_MAX;

  NFA=bnew.nfa;
    
  free_ntuple_list(bnew.l);
  free(cells);
  return NFA;

}

/*  free alignment_list structure */
void free_alignment_list(alignments_list *a){
  // printf("free_alignment_list started \n");
  int i;
  for(i=0;i<a->pos;i++) {
    // printf("free_alignment_list loop %d actual size %d, array size:%d\n", i, (int)a->pos,sizeof(a->array)/sizeof(a->array[0]));
    free_ntuple_list((a->array[i]).l);
  }
  // printf("free_alignment_list loop finished\n\n\n");
  
  // printf("freeing a->array \n");
  free(a->array);
  // printf("freeing a itself \n");
  free(a);
}
