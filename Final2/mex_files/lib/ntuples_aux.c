#include "ntuples_aux.h"



/*----------------------------------------------------------------------------*/
/* append value to ntuple list */
void append_to_ntuple( ntuple_list a, double x){
  if (a->size == a->max_size) enlarge_ntuple_list(a);
  a->values[(a->size)*(a->dim)]=x;
  a->size++;
}


/*----------------------------------------------------------------------------*/
/* copy ntuple */

ntuple_list copy_ntuple_list(ntuple_list  a){
  /* copies an ntuple of dimension 1 or 2 */
  ntuple_list b;
  b=new_ntuple_list(a->dim);

  int i;
  double x,y;

  for (i=0;i<a->size;i++){
    if (a->dim==2){

      x=a->values[i*a->dim+0];
      y=a->values[i*a->dim+1];

      if( b->size == b->max_size ) enlarge_ntuple_list(b);
      b->values[ b->size * b->dim + 0 ] = x;
      b->values[ b->size * b->dim + 1 ] = y;
      b->size++;

    }

    if(a->dim==1){
      x=a->values[i];
      if( b->size == b->max_size ) enlarge_ntuple_list(b);
      b->values[ b->size ] = x;
      b->size++;

    }

  }

  return b;

}

/* ---------------------------------------------------------------------------*/
/* Check if a 1D-ntuple contains an item */
int is_in_ntuple(ntuple_list l, double x){
  int i;
  for (i=0; i<l->size;i++){
    if (l->values[i]==x) return 1;
  }
  return 0;
}


/*----------------------------------------------------------------------------*/
/** Read TXT file with point coordinates.
 */
ntuple_list read_points(char * filename)
{
  ntuple_list p = new_ntuple_list(2);
  FILE * f;
  double x,y;
  int n,m;

  /* open file */
  f = fopen(filename,"r");
  if( f == NULL ) error("unable to open input file.");

  /* read points */
  do
    {
      n = fscanf(f,"%lg",&x);
      m = fscanf(f,"%lg",&y);

      if( n==1 && m==1 )
        {
          if( p->size == p->max_size ) enlarge_ntuple_list(p);
          p->values[ p->size * p->dim + 0 ] = x;
          p->values[ p->size * p->dim + 1 ] = y;
          p->size++;
        }
    }
  while( n==1 && m==1 );

  /* close file if needed */
  if( fclose(f) == EOF )
    error("unable to close input file.");

  /* return list of points */
  return p;
}


