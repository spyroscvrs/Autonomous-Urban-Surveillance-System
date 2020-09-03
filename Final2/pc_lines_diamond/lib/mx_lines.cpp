#include <list>
#include <stdio.h>
//#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#define ELLIPSE_THRESH 32

//neighborhoods
const int xinc[8] = {-1,-1,-1, 0,0, 1,1,1};
const int yinc[8] = {-1, 0, 1,-1,1,-1,0,1};

template <class T> const T& max (const T& a, const T& b) {
  return !(b>a)?a:b; 
}

struct line_param
{
    float a;
    float b;
    float c;
    float w;
    line_param(float _a,float _b, float _c, float _w):a(_a),b(_b),c(_c),w(_w){}
};

struct coords
{
    int x;
    int y;
    int d;
    coords(int _x,int _y, int _d):x(_x),y(_y),d(_d){}
};

std::list<coords> floodFill(int *data, int cx, int cy, int img_height, int * rads, int rads_size, int * v_steps, int * dist);
int fit_ellipse(std::list<coords> & results, float * vec_data);
int eigens(float * cov_data, float * vec_data);

extern "C" void mexFunction(int * data, int width, int height, int * rads, int rads_size,float ** mx_lines_data, int * lines_num);
void mexFunction(int * data, int width, int height, int * rads, int rads_size, float ** mx_lines_data, int * lines_num)
{
//    printf("entered edgelet content first 10:\n");
//    for(int i =0;i < height;i++){
//        for(int j = 0;j< width;j++){
//            printf("%d ", data[i * height + j]);
//        }
//        printf("\n");
//    }
//    printf("\n");
    //init matrix
//    int * data = (int*)mxGetData(prhs[0]); 
      
//    int width = int(mxGetN(prhs[0]));
//    int height = int(mxGetM(prhs[0]));
    
    //rings radius
//    int * rads = (int*)mxGetData(prhs[1]); 
//    int rads_size = int(mxGetN(prhs[1]));
//    printf("getting patch size\n");
    int patch_size = rads[rads_size-1]*2 + 1;
//    printf("patch size %d", patch_size);
//    printf("vseps initialization\n");
    //v_step for faster indexing
    
    int * v_steps = (int*)malloc(sizeof(int)*patch_size);
    for(int i = 0; i < patch_size; i++) 
        v_steps[i] = i*patch_size;
    
//    printf("initializing dist array\n");
    //mask with distances
    int * dist = (int*)malloc(sizeof(int)*patch_size*patch_size);
//    printf("distances:::\n");
    for(int i=0; i<patch_size; i++){
        for(int j=0; j<patch_size; j++){
            dist[v_steps[i] + j] = max(abs(i-rads[rads_size-1]), abs(j-rads[rads_size-1]));
//            printf("%d ", dist[v_steps[i] + j]);
        }
//        printf("\n");
    }
//    printf("\n");
    std::list<line_param> lines; 
    
    //normalize values
    float w_c = (width - 1)/2.f;
    float h_c = (height - 1)/2.f;
    float norm = (max(w_c, h_c) - rads[rads_size-1]);
//    printf("wc h_c norm %f %f %f \n", w_c, h_c, norm);
    float vec_data[4];
//    printf('')
//    printf("main process started\n");
    //for each edge point in image with padds
    for(int i=rads[rads_size-1]; i <  width - rads[rads_size-1]; i++)
    for(int j=rads[rads_size-1]; j <  height - rads[rads_size-1]; j++)
    {
//        printf('')
        if (data[i * height + j] == 1)
        {
//        printf("flood fill \n");
//            printf("starting to find lines\n");
            //call flood fill with array of rings radius    
            std::list<coords> results = floodFill(data, i, j, height, rads, rads_size, v_steps, dist);
//          	printf("starting to fit an ellipse\n");
//            printf("fitt ellipse\n");
            //fit ellipse
            if(fit_ellipse(results, vec_data) == 1)
            {
                //push if ok                
                float c = -vec_data[0]*((j - h_c)/norm) - vec_data[1]*((i - w_c)/norm); 
//                printf("line params %f, %f, %f \n", vec_data[1], vec_data[0], c);
                lines.push_back(line_param(vec_data[1], vec_data[0], c, 1.f));      
            }
        }  
    }
//    printf("main process is finished\n");
    
//    printf("mxlines initializatino\n");
//    int lines_size[2] = {4, lines.size()};
//    mxArray * mx_lines = mxCreateNumericArray(2, lines_size, mxSINGLE_CLASS, mxREAL);
//    float * mx_lines_data = (float*)mxGetData(mx_lines); 
   
    float * mx_lines = (float*)malloc(sizeof(float) * 4 * lines.size());
    int c = 0;
    for (std::list<line_param>::iterator it = lines.begin(); it!=lines.end(); ++it, c++)
    {
        mx_lines[c*4] = it->a;               
        mx_lines[c*4 + 1] = it->b;               
        mx_lines[c*4 + 2] = it->c;
        mx_lines[c*4 + 3] = it->w;
    }
    *mx_lines_data = mx_lines;
    //plhs[0] = mx_lines;
    //free(labData);
    //plhs[0] = idx;
    int num_lines_temp= lines.size();
//    printf("returning lines num = %d \n",num_lines_temp);
    *lines_num = lines.size();
//    printf("freeing vsteps and dist\n");
    free(v_steps);
    free(dist);    
}


std::list<coords> floodFill(int *data, int cx, int cy, int img_height, int * rads, int rads_size, int * v_steps, int * dist)
{   
    std::list<coords> points;    
    std::list<coords> results;  
    
    int patch_size = rads[rads_size-1]*2 + 1;
    int radius = rads[rads_size-1];
    points.push_back(coords(radius, radius, 0));
    
    int * labels = (int*)calloc(patch_size*patch_size, sizeof(int));        
    int * hist = (int*)calloc(radius+1, sizeof(int));
//    printf("radius %d\n",radius );
//    printf("patch_size %d\n",patch_size);
//    printf("hist first 100:\n");
//    for(int i =0;i < 10;i++){
//         printf("%d ", hist[i]);
//     }   
//     printf("\n");
    
    hist[0] = 1;
    
    int eque_min = 0;
    
    coords C(radius, radius, 0);
    
    int x_offset = cx - radius;
    int y_offset = cy - radius;
    
    for(int i=0; i<rads_size; i++)
    {
        while (points.size()!=0 && eque_min <= rads[i])
        {               
            for (std::list<coords>::iterator j = points.begin(); j!=points.end(); ++j)
            {
                if(j->d == eque_min)
                {
                    C = (*j); 
                    points.erase(j);
                    break;
                }                
            }
                        
            hist[C.d]--;
                       
            if(labels[v_steps[C.x] + C.y] == 0)
            {
                labels[v_steps[C.x] + C.y] = i+1;
                results.push_back(coords(C.y - radius, C.x - radius, i+1));
    
                for(int j=0; j<8; j++)
                {
                    int x = C.x + xinc[j];
                    int y = C.y + yinc[j];
                    if(x >= 0 && x < patch_size && y >=0 && y < patch_size && data[(x + x_offset)*img_height + y + y_offset] == 1 && labels[v_steps[x] + y] == 0)
                    {
                        int d = dist[v_steps[x] + y];
                        points.push_back(coords(x, y, d));
                        hist[d]++;
                    }
                }
            }   
            
            for(int j=0; j<radius + 1; j++)
            {
                if(hist[j]!=0) 
                {
                    eque_min = j;
                    break;
                }
            }            
        }
    }    
    
    free(hist);
    free(labels);
    return results;
}

int fit_ellipse(std::list<coords> & results, float * vec_data)
{
    int good_ellipse = 0;
    
    int dist = results.back().d;
    
    while (good_ellipse == 0 && dist > 0)
    {
        float cov_data[3];            
        
        //get mean
        float x_mean = 0;
        float y_mean = 0;    
    
        int size = 0;
        for (std::list<coords>::iterator it = results.begin(); it!=results.end(); ++it)
        {
            if(it->d > dist) break;
            x_mean += it->x;
            y_mean += it->y;            
            size++;
        }
    
        if(size < 3) return good_ellipse;
        
        x_mean /= size;
        y_mean /= size;         
            
        //get covariance
        float xx = 0;
        float yy = 0;
        float xy = 0;
    
        for (std::list<coords>::iterator it = results.begin(); it!=results.end(); ++it)    
        {
            if(it->d > dist) break;
            float x = it->x - x_mean;
            float y = it->y - y_mean;
            xx += x*x;
            yy += y*y;
            xy += x*y;            
        }    
    
        cov_data[0] = xx / (size - 1);
        cov_data[1] = xy / (size - 1);
        cov_data[2] = yy / (size - 1);
    
        good_ellipse = eigens(cov_data, vec_data);
        dist--;
    }
    
    return good_ellipse;
}
 
int eigens(float * cov_data, float * vec_data)
{
    float trace = cov_data[0] + cov_data[2];
    float det = cov_data[0]*cov_data[2] - cov_data[1]*cov_data[1];
    
    float s = sqrt(trace*trace/4 - det);
    float vals_data[2] = {trace/2 - s, trace/2 + s};
        
   if(abs(cov_data[1]) > 0.001 )
    {
        vec_data[0] = vals_data[0] - cov_data[2];        
        vec_data[1] = cov_data[1];
        
        float n = sqrt(vec_data[0]*vec_data[0] + vec_data[1]*vec_data[1]);
        vec_data[0] /= n;
        vec_data[1] /= n;
    }else
    {
        vec_data[0] = int(cov_data[0] < cov_data[2]);
        vec_data[1] = vec_data[0] - 1;
    }
    
    vec_data[2] = vec_data[1];
    vec_data[3] = -vec_data[0];
    
    return (abs(vals_data[0]) < 0.001 || vals_data[1]/vals_data[0] > ELLIPSE_THRESH);
}
