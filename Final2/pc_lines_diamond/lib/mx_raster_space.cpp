#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <string.h>
void lines_end_points(float * line, int * endpoints, float space_c, int numLines);
void rasterize_lines(float * line, int * endpoints, int * space, int SpaceSize, int numLines);
inline void lineH(int x0, int y0, int x1, int y1, int * space, int * y_steps, int weight);
inline void lineV(int x0, int y0, int x1, int y1, int * space, int * y_steps, int weight);

template <typename T> int sgn(T val) 
{
    return (T(0) <= val) - (val < T(0));
}

int round(float x) 
{
    return ((x>=0) ? int(x + 0.5) : int(x - 0.5));
}
//extern "C" void mexFunction1();
//void mexFunction1(){
//    printf("\nspaceSize should be two dim array size1111111111\n\n");
//}
extern "C" void free_int_array(int **input);
void free_int_array(int ** input)
{
    free(input);
}

extern "C" void mexFunction(float * LinesData, int * SpaceSize, int numLines, int ** pSpace_out);
void mexFunction(float * LinesData, int * SpaceSize, int numLines, int ** pSpace_out)
{


//    printf("newSpaceSize= %d ,%d\n",SpaceSize[0] , SpaceSize[1] );
//    printf("numLines= %d\n",numLines);
    int * pSpace = new int[SpaceSize[0] * SpaceSize[1]]();//(int*) malloc(sizeof(int) * SpaceSize[0] * SpaceSize[1]);
    
//    memset(pSpace, 0, sizeof(pSpace));
    //Get Lines data
    float space_c = (SpaceSize[0] - 1.f)/2;

    //int cE[2] = {8, numLines};
    //mxArray * mxE = mxCreateNumericArray(2, cE, mxINT32_CLASS, mxREAL);
    //int * EndPoints = (int*)mxGetData(mxE); 
    int * EndPoints =  new int[8*numLines]();//(int*) malloc(sizeof(int)*8*numLines);
    printf("EndPoints[0]= %d\n",EndPoints[0]);
//    printf("endpoints allocated\n");
    
//    
//    printf("contents of LinesData 12 ta\n");
//    for(int i = 0; i < 12; i++)
//    {
//        printf("%f ",LinesData[i]);
//    }
//    
//    printf("getting all line endpoints\n");
    //Get all EndPoints
    lines_end_points(LinesData, EndPoints, space_c, numLines);
    
    
    
//    
//    printf("lines end points found \n");
//    printf("ouput contents before\n");
//    for(int i = 0; i < 10; i++)
//    {
//        printf("%d ",pSpace[i]);
//    }
    
//    printf("rasterize lines started\n");
//    Rasterize
    rasterize_lines(LinesData, EndPoints, pSpace, SpaceSize[0], numLines);
    
//    printf("freeing endpoints\n");
//    printf("rasterize lines finishied \n");
    free(EndPoints);
//    printf("ouput contents \n");
//    for(int i = 0; i < 10; i++)
//    {
//        printf("%d ",pSpace[i]);
//    }
//    printf("returning data\n");
    *pSpace_out = pSpace;
//    for(int i =0;i < SpaceSize[0];i++){
//        for(int j = 0;j < SpaceSize[1];j++){
//            printf("%d ",pSpace[i * SpaceSize[1] + j]);
//        }
//        printf("\n");
//    }
//       printf("\n");
//    free(pSpace); // first pointer must be released
}

void rasterize_lines(float * line, int * endpoints, int * space, int cSpaceSize, int numLines)
{ 
//    printf("rasterize lines started \n");
    int * v_steps = (int*)malloc(sizeof(int)*cSpaceSize);
    for(int i = 0; i < cSpaceSize; i++) 
        v_steps[i] = i*cSpaceSize;
//    printf("v steps intialized \n");
    
    for(int i = 0; i < numLines; i++)
    {
//    printf("loop %d\n", i);
        int * end = endpoints + i*8;
//        printf("loop end init end\n");    
        int weight =  int(*(line + i*4 + 3));
//        printf("loop weights init end\n");
        for(int j=0; j<6; j+=2)
        {
//            printf("inner loop %d\n", j);
            if(abs(end[j+3] - end[j+1]) > abs(end[j+2] - end[j])){
//                printf("inner loop linev started \n");
                lineV(end[j], end[j+1], end[j+2], end[j+3], space, v_steps, weight);
//                printf("inner loop linev finished \n");
                }
            else{
//                printf("inner loop lineH started with values x1=%d, y1=%d x2=%d,y2=%d\n", end[j], end[j+1], end[j+2], end[j+3]);
                lineH(end[j], end[j+1], end[j+2], end[j+3], space, v_steps, weight);        
//                printf("inner loop lineH finished \n");
            }
        }
//        printf("loop inner loop end\n");
        space[v_steps[end[7]] + end[6]] += weight;
//        printf("loop weight added end\n");
    }
//    printf("freeing vsteps in rasterize lines method \n");
    free(v_steps);
}

void lines_end_points(float * line, int * endpoints, float space_c, int numLines)
{
    int center = round(space_c);
    for(int i = 0; i < numLines; i++)
    {
        float a = *(line + i*4);
        float b = *(line + i*4 + 1); 
        float c = *(line + i*4 + 2);
    
        float alpha = float(sgn(a*b));
        float beta = float(sgn(b*c));
        float gamma = float(sgn(a*c));
    
        int * end = endpoints + i*8;

        float a_x = alpha*a / (c + gamma*a);
        float b_x = -alpha*c / (c + gamma*a);

        end[1] = round((a_x + 1) * space_c); //that is why the output is always positive or zero
        end[0] = round((b_x + 1) * space_c); 

        end[3] = round((b / (c + beta*b) + 1) * space_c);
        end[2] = center;

        end[5] = center;
        end[4] = round((b / (a + alpha*b) + 1) * space_c);

        end[7] = round((-a_x + 1) * space_c);
        end[6] = round((-b_x + 1) * space_c);    
    }
}

inline void lineH(int x0, int y0, int x1, int y1, int * space, int * y_steps, int weight)
{
//    printf(" line h started\n");
//    printf("getting the slope\n");
    float slope = (float)(y1 - y0)/(x1 - x0);

    
	//float y_iter = y0 + 0.5f;     
    float y_start = y0 + 0.5f; 
    float y_iter = y_start;
    
    int step = (x0 < x1) ? 1 : -1;
    slope *= step;
    
    
    for(int x = x0, c = 1; x != x1; x+=step, c++)
	{   
//    	printf("line h inner loop from x0=%d to x1=%d, curr x = %d\n",x0,x1,x);
        space[y_steps[int(y_iter)] + x] += weight;        
        //y_iter += slope;		  		
        y_iter = y_start + c*slope;
	}
}

inline void lineV(int x0, int y0, int x1, int y1, int * space, int * y_steps, int weight)
{
    float slope = (x1 - x0)/(float)(y1 - y0);

	//float x_iter = x0 + 0.5f; 
    float x_start = x0 + 0.5f; 
    float x_iter = x_start;
    int step = (y0 < y1) ? 1 : -1;
    slope *= step;

    for(int y = y0, c = 1; y != y1; y+=step, c++)
	{	
        space[y_steps[y] + int(x_iter)] += weight;
        //x_iter += slope;		  
        x_iter = x_start + c*slope;        
	}     
}
