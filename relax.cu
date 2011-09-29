#ifndef _RELAX_H_
#define _RELAX_H_
//#include <cutil_inline.h>
//extern "C" 
#include "vertex.h"
/*
struct edge{
    int des_v;
    int distance;
};

struct vertex{
    struct edge *edges;
    int dist;
    int pre_vertex;
};

struct gpuResult{
  int index;
  int old_distance;
  int new_distance;
};
*/
__global__ void
relax_all(int* gpu_vertex_buf, cpu::vertex* gpu_global_vertex, cpu::edge* gpu_global_edge, cpu::gpuResult* cpu_result, cpu::gpuResult* gpu_used_result_buf){
    const unsigned int bid = blockIdx.x; 
    const unsigned int num_block = gridDim.x; 
    const unsigned int tid_in_block = threadIdx.x;
    const unsigned int num_thread = blockDim.x;
    const unsigned int tid_in_grid = blockDim.x * blockIdx.x +threadIdx.x;

    //cpu::gpuResult* gpu_result_buf;
    //cudaMalloc((void **)&gpu_result_buf, sizeof(cpu::gpuResult)*num_block*num_thread);  

    int i=0,j=0,num_edges=0,tent_current=0;
    for (i=bid;i<1024;i+=num_block){
        cpu::vertex *temp_v = &gpu_global_vertex[gpu_vertex_buf[i]];
        num_edges = gpu_global_vertex[i+1].edges -temp_v->edges;
        tent_current = temp_v->dist;
        for(j=tid_in_block;j<num_edges;j+=num_thread){
            int dist_current = (temp_v->edges+j)->distance;
            int dest = (temp_v->edges+j)->des_v;
            int tent_dest = gpu_global_vertex[dest].dist;
            int flag = 0;
            if(tent_current + dist_current < tent_dest)
                flag =1;
                
            gpu_used_result_buf[tid_in_grid].index = dest*flag;
            gpu_used_result_buf[tid_in_grid].old_distance = tent_dest*flag;
            gpu_used_result_buf[tid_in_grid].new_distance = (tent_current+dist_current)*flag;
            
            }
        //sync for sent result back
        //FIXME
        //__syncthreads();
       //cudaMemcpy((cpu_result+blockDim.x * blockIdx.x),gpu_used_result_buf, sizeof(cpu::gpuResult)*num_thread, cudaMemcpyDeviceToHost);

       //cutilSafeCall(cudaMemcpy((cpu_result+blockDim.x * blockIdx.x),gpu_result_buf, sizeof(cpu::gpuResult)*num_thread, cudaMemcpyDeviceToHost));
    }
    }
    //cutilSafeCall(cudaFree(gpu_result_buf));  

#endif
