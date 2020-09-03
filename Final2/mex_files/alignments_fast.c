/*-----------------------------------------------------------------------------
alignments_fast                                                                \
author: jose lezama/rafael grompone von gioi                                   \
version: 8.1 (2014.08.12)                                                      \
year: 2014                                                                     \
desc: MatLab interface for aligned point detector v8 (w/candidate points).     \
input:                                                                         \
-req: input_points | Input points (Nx2 matrix).                                \
-req: candidate_points | Candidate points for alignments (Nx2 matrix).         \
-opt: min_k |  Minimum number of points in alignment.    		       \
-opt: xsize | X axis domain size.                        		       \
-opt: ysize | Y axis domain size.                        		       \
-opt: epsilon | Detection threshold, (max. NFA).             		       \
-opt: min_width |  Minimal alignment width tested.       		       \
-opt: locality |  Size factor of locality.               		       \
-opt: length_ratio |  Min ratio b/w length and width.    		       \
output:                                                                        \
-6xNout alignments matrix with: x1, x2, y1, y2, width, nfa                     \
------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include "lib/jirafa_lib.h"
#include "lib/ntuples_aux.h"
#include "lib/alignments.h"
/** sqrt(2) */
#define SQRT2    1.414213562373

/* debug function */
#define dprint(expr) printf(#expr " = %g \n", (double) expr)


/*----------------------------------------------------------------------------*/
/*                             look for alignments                            */
/*                           (core of the algorithm)                          */
void detect_alignments(double logNT, double log_epsilon, ntuple_list points,
		       ntuple_list candidate_points,
                       double min_width, double locality, double length_ratio,
                       double min_k, int * cells,
                       alignments_list *all_alignments, 
                       struct point_alignment *best_align){


  int i,j,l,n;
  double x,y,xc,yc,dx,dy,theta,lateral_dist,long_dist,fwd_dist;
  double width;
  struct point_alignment align;
  ntuple_list ll;


  /* initialize best_align */
  best_align->nfa = -logNT; // -DBL_MAX;
  best_align->l = NULL;
  

  int NlocalLeft = 0;
  int NlocalRight = 0;
  int NlocalAlignment = 0;
  int point_in_alignment=0;


  /* test all possible candidate pairs*/

  time_t mytimer;
  mytimer = time(NULL);
  int total_num_tests = 0;

#pragma omp parallel for schedule(dynamic) private(i, j, ll, align, l, n, x, y, xc, yc, dx, dy, theta, locality, lateral_dist, long_dist, fwd_dist, width, NlocalLeft, NlocalRight, NlocalAlignment, point_in_alignment, cells) shared(best_align, all_alignments, mytimer, total_num_tests) 

  for(i=0; i<candidate_points->size; i++) 
    for(j=i+1; j<candidate_points->size; j++)

      {
        /* if ( (int) (time(NULL)-mytimer) >3){ */
        /*   printf("looking for alignments i: %d, thread: %d \n", i, omp_get_thread_num()); */
        /*   mytimer = time(NULL); */
        /* } */
        
        align.x1 = candidate_points->values[ i*candidate_points->dim + 0 ];
        align.y1 = candidate_points->values[ i*candidate_points->dim + 1 ];
        align.x2 = candidate_points->values[ j*candidate_points->dim + 0 ];
        align.y2 = candidate_points->values[ j*candidate_points->dim + 1 ];
        align.len = dist(align.x1,align.y1,align.x2,align.y2);

        if (align.len < length_ratio) continue;

        xc = ( align.x1 + align.x2 ) / 2.0;
        yc = ( align.y1 + align.y2 ) / 2.0;
        theta = atan2( align.y2 - align.y1, align.x2 - align.x1 );
        align.dx = dx = cos(theta);
        align.dy = dy = sin(theta);


        int wi;
        width =  (align.len / length_ratio);

        int * cells2;
        cells2 = (int *) calloc( (size_t) 1000, sizeof(int) );


        for(wi = 0; wi <6 ; wi++)
          {
	    width/=2;//SQRT2;
            


            if(width < min_width) break;
            

            align.width = width;

            /* use around K cases where K is the number of points in the
               alignment */
            /* count number of points in alignment */
            double K=0;
            for(l=0;l<points->size;l++){
	      // TODO remove endpoints
	      x = points->values[ l*points->dim + 0 ];
	      y = points->values[ l*points->dim + 1 ];
	      lateral_dist = fabs( -(x-xc)*dy + (y-yc)*dx );
	      fwd_dist = (x-align.x1)*dx + (y-align.y1)*dy;
	      /* the point is in the alignment? */
	      if( lateral_dist < 0.5 * width &&
		  fwd_dist > 0 &&
		  fwd_dist < align.len )
		K++;
	    }

            if (K < min_k) {
              
              continue;
            }

            /* explore a range of number of cases around K */
            double min_n = (int) fmax(4,K/2);
            double max_n = (int) 1.5*K;

            for(n = min_n; n<=max_n; n=(int) n*SQRT2)
              {
                

                align.n = n;
                align.s = align.len/(double) (n+1);   /* cell size */

                double loc_exp;
                
                // 25 sep 2013, begin with square local window
                locality = (align.len - width)/(2.0*width)*SQRT2;

                //for(loc_exp = 2; loc_exp <=32; loc_exp *= 2 ){
                for(loc_exp = 0; loc_exp <6; loc_exp++){

                  //locality = pow(2,loc_exp/2.0); 
                  locality/=2; //SQRT2;

                  if (width*locality>250 || locality < 2) break;


                  /* po is the probability of 1 point falling in one case */
                  align.po = 1.0 / ( (double) n * (2.0*locality+1.0) );

                  /*     /\* compute active cells and number of local points*\/ */
                  align.k = 0;
                  for(l=0;l<n;l++) cells2[l] = 0;
                  align.locality = locality;
                  align.Nlocal = 0;

                  /* new in algorithm 3: count points left and right of the
                     alignment */
                  NlocalLeft = NlocalRight = NlocalAlignment  = 0;
		
                  /* initialize list of interior points */
                  ll = new_ntuple_list(1);
                  align.l = ll;// new_ntuple_list(1);
                
                  for(l=0;l<points->size;l++)
		    // TODO: remove endpoints
		    {
		      x = points->values[ l*points->dim + 0 ];
		      y = points->values[ l*points->dim + 1 ];
		      point_in_alignment = 0;

		      lateral_dist = fabs( -(x-xc)*dy + (y-yc)*dx );
		      fwd_dist = (x-align.x1)*dx + (y-align.y1)*dy;
		      long_dist = fabs( (x-xc)*dx + (y-yc)*dy );
		      

		      /* the point is in the alignment? */
		      if( lateral_dist < 0.5 * width &&
			  fwd_dist > 0.5 * align.s &&
			  fwd_dist < ( (double) n + 0.5 ) * align.s )
			{
			  point_in_alignment=1;
			  NlocalAlignment++;
			  cells2[ (int) (fwd_dist/align.s - 0.5) ] = 1;
			  /* point is in the alignment */
			  /* append the point */
			  append_point_to_alignment(&align, (unsigned int) l);
			  /* align.k++; */
			}
		      /* the point is in the local area? */
		      if( lateral_dist < (locality+0.5)*width &&
			  long_dist < ( (double) n / 2.0 *align.s ) )
			{
			  align.Nlocal++;
			  if(point_in_alignment <1)
			    if (-(x-xc)*dy + (y-yc)*dx >0)
			      NlocalLeft++;
			    else
			      NlocalRight++;
			}

		    }

                  /* count occupied cases */
                  for(l=0;l<n;l++) align.k += cells2[l];


                  if (align.k <=0) {
                    free_ntuple_list(align.l);
                    continue;
                  }
                  /* /\* compute event probability *\/ */

                  align.Nlocal = 2*fmax(NlocalLeft,NlocalRight) + NlocalAlignment;
                  /* dprint(align.Nlocal); */
                  
                  align.p = 1.0 - pow( 1.0 - align.po, (double) align.Nlocal );
                  if( align.p > 0.0 && align.p < 1.0 ){
                    align.nfa = nfa(n,align.k,align.p,logNT);
                    total_num_tests++;
                  }

                  else {
                    align.nfa =  -logNT; // -DBL_MAX;
                    total_num_tests++;

                  }

#pragma omp critical
                  {

		    if( align.nfa > best_align->nfa ) {
		      if(best_align->l!=NULL) free_ntuple_list(best_align->l);
		      copy_align(&align,best_align);
		    }
		    if( align.nfa >= log_epsilon ) /* if significant event */
		      /*if( 1 ) /* if significant event */
		      {
			append_alignment(all_alignments, &align);

		      }
                  }

                  free_ntuple_list(align.l);

                }
              }
          }
        free(cells2);
      }
  
  //printf("total real number of tests: %d\n", total_num_tests);
}

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*                                    Main                                    */
/*----------------------------------------------------------------------------*/
int main(int argc, char ** argv)
{
  return 0;
}

/*----------------------------------------------------------------------------*/
/*                                    Matlab interface                        */
/*----------------------------------------------------------------------------*/

extern void  mexFunction_alignment_fast(double * input_points, int X_in, int N_in,double * input_candidate_points,int NCP_in, double ** output_points, int *X_out, int *N_out);
void  mexFunction_alignment_fast(double * input_points, int X_in, int N_in,double * input_candidate_points,int NCP_in, double ** output_points, int *X_out, int *N_out){
  /* 
     input: list of 2D points and candidates
     output: 7xN matrix with info for N alignment detections
  */

  double xsize;
  double ysize;
  double epsilon;
  double min_width;
  double locality;
  double min_k;
  double length_ratio;

  
  min_k = 5;
  xsize = 512;
  ysize = 512;
  epsilon = 10;
  min_width = 1;
  locality = 10;
  length_ratio = 25;


  /* variable declarations here */
  // double * input_points = mxGetPr(prhs[0]); // input points of size 2*N first N
					    // x coordinates, then N y
					    // coordinates
  // double * input_candidate_points = mxGetPr(prhs[1]); // input points of size 2*N first N
  // x coordinates, then N y
  // coordinates



  int X = X_in; //number of columns (should be 2)
  int N = N_in; // number of  points
  int NCP = NCP_in; // number of candidate points


  omp_set_num_threads(256);


  struct point_alignment align, best_align;

  /* initiate alignements list */
  alignments_list *all_alignments = (alignments_list*)calloc(sizeof(alignments_list), 1);
  

  all_alignments->array = (struct point_alignment*)realloc(all_alignments->array, (all_alignments->capacity = 4) * sizeof(struct point_alignment));


  ntuple_list points;
  ntuple_list candidate_points;


  double x,y,xc,yc,dx,dy,theta,lateral_dist,long_dist,fwd_dist;
  double width;
  unsigned int i,j,l,n;
  int num_test = 0;
  double logNT;
  int * cells;
  FILE * eps;

  time_t timer;

  /* read input */
  points = new_ntuple_list(2);
  for(i=0;i<N;i++){
    if( points->size >= points->max_size ) enlarge_ntuple_list(points);
    points->values[ points->size * points->dim + 0 ] = input_points[i];
    points->values[ points->size * points->dim + 1 ] = input_points[N+i];
    points->size++;
  }

  candidate_points = new_ntuple_list(2);
  for(i=0;i<NCP;i++){
    if( candidate_points->size >= candidate_points->max_size ) enlarge_ntuple_list(candidate_points);
    // printf("candidate_point_values[%d]=[%f %f]\n", i , input_candidate_points[i], input_candidate_points[NCP+i]);
    candidate_points->values[ candidate_points->size * candidate_points->dim + 0 ] = input_candidate_points[i];
    candidate_points->values[ candidate_points->size * candidate_points->dim + 1 ] = input_candidate_points[NCP+i];
    candidate_points->size++;
  }




  n=points->size;

  /* NUMBER OF TESTS */
  num_test= n * (n-1)/2; /* widhts are ignored */


  //printf("points: %d\n", n);
  //printf("num_test: %d\n", num_test);
  logNT = log10( (double) num_test );
  double log_epsilon = log10(epsilon);

  

  /* initialize best_align */
  best_align.nfa = -logNT;


  int nmax = dist(xsize,ysize,0.0,0.0) / min_width;
  cells = (int *) calloc( (size_t) nmax, sizeof(int) );
  if( cells == NULL ) error("not enough memory.");


  /*---------------  detect alignments -------------------------------------- */
  //printf("detecting alignments...\n");
  timer = time(NULL);

  detect_alignments(logNT, log_epsilon, points, candidate_points, min_width, locality, length_ratio, 
                    min_k, cells, all_alignments, &best_align);

  //printf("\nfinished detecting alignments...\n");
  //printf("time elapsed: %d seconds\n", (int) (time(NULL)-timer));
  dprint(best_align.nfa);

  /*--------------- end detect alignments------------------------------------ */



  /* sort alignements */
  qsort ((*all_alignments).array, all_alignments->pos,
         sizeof(struct point_alignment), compare_alignments);
  /* --------------------------------------------------------------------------------- */
  /* --------------------------------------------------------------------------------- */
  /* 1-vs-1 exclusion principle: start a list of alignments with the
     most significant one, then one by one check if they are masked */
  /* IMPORTANT: all_alignments are ordered backwards */
  /* create the list */

  //printf("applying exclusion principle on resulting %d alingments...\n", all_alignments->pos);
  timer = time(NULL);

  alignments_list *f1v1_alignments = (alignments_list*)calloc(sizeof(alignments_list), 1);

  /* it is possible that there are no alignments*/
  if (all_alignments->pos != 0){

    /* append the first alignment to this list */
    append_alignment(f1v1_alignments, &(all_alignments->array[all_alignments->pos-1]));



    int ismasked;  
    signed int ai;

    for(ai=(*all_alignments).pos-2;ai>=1;ai--){
      struct point_alignment temp_alignment_b = all_alignments->array[ai];

      ismasked=0;

      for(j=0;j<(*f1v1_alignments).pos;j++){
        struct point_alignment temp_alignment_a = f1v1_alignments->array[j];

        /* check if alignments from final list mask alignment in
           all-alignments list */

        double im=is_masked_cases(points, &temp_alignment_a, 
                                  &temp_alignment_b, logNT);

        if (im<=log_epsilon) ismasked=1;
      }

      if (ismasked ==0){		/* not masked, add to final list */
        append_alignment(f1v1_alignments, &temp_alignment_b);
      }
    }

    //printf( "number of final alignments after 1vs1: %d \n",(int) (*f1v1_alignments).pos);
  }

  //printf("\nfinished exclusion principle...\n");
  //printf("time elapsed: %d seconds\n", (int) (time(NULL)-timer));

  /* END of 1 vs 1 exclusion principle */
  /* ------------------------------------------------------------------------ */

  /*----------------- MATLAB:  prepare output variables --------------------*/

  /* output variables definitions */
  int lw = 6; // x1,y1,x2,y2,width,-log10(NFA)
  int n_out = (*f1v1_alignments).pos;
  // plhs[0] = mxCreateDoubleMatrix( lw, n_out ,mxREAL);



  //Get a pointer to the data space in our newly allocated memory
  double* outArray = (double *) malloc (sizeof (double) * (lw * n_out));

  //Copy matrix while multiplying each point by 2

  int ii;
  for( ii=(*f1v1_alignments).pos-1;ii>=0;ii--){ /*  hack to do
                                                    backward loop */

    i=ii;

    struct point_alignment temp_alignment = f1v1_alignments->array[i];

    outArray[ii*lw+0] = temp_alignment.x1;
    outArray[ii*lw+1] = temp_alignment.y1;
    outArray[ii*lw+2] = temp_alignment.x2;
    outArray[ii*lw+3] = temp_alignment.y2;
    outArray[ii*lw+4] = temp_alignment.width;
    outArray[ii*lw+5] = temp_alignment.nfa;

  }

  /*--------------------------- end of MATLAB -------------------------------*/


  /* free memory */

  if(best_align.l != NULL) free_ntuple_list(best_align.l);
  free_alignment_list(all_alignments);
  free_alignment_list(f1v1_alignments);

  free(cells);
  free_ntuple_list(points);

  *output_points = outArray;
  *X_out = 6;
  *N_out = n_out;
  /*
    free( (void*) cells );
    free_ntuple_list(points);
  */
}
/*----------------------------------------------------------------------------*/



