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
get_result(cpu::vertex* gpu_global_vertex,int i){
	printf("result: %d\n",gpu_global_vertex[i].dist);	
}

__global__ void
relax_all(int* gpu_vertex_buf, cpu::vertex* gpu_global_vertex, cpu::edge* gpu_global_edge, cpu::gpuResult* cpu_result, cpu::gpuResult* gpu_used_result_buf){
#ifdef DEBUG
    /*for(int i = 0; i < MAX_BUCKET_SIZE; i++){
        if(gpu_vertex_buf[i] != 0){
            printf("GPU %d: %d\n", i, gpu_vertex_buf[i]);
        }
    }*/
#endif 
    const unsigned int bid = blockIdx.x; 
    const unsigned int num_block = gridDim.x; 
    const unsigned int tid_in_block = threadIdx.x;
    const unsigned int num_thread = blockDim.x;
    const unsigned int tid_in_grid = blockDim.x * blockIdx.x +threadIdx.x;

#ifdef DEBUG
    //printf("%d, %d, %d, %d, %d\n", bid, num_block, tid_in_block, num_thread,
    //        tid_in_grid);
#endif
    //cpu::gpuResult* gpu_result_buf;
    //cudaMalloc((void **)&gpu_result_buf, sizeof(cpu::gpuResult)*num_block*num_thread);  

    int i=0,j=0;//,num_edges=0,tent_current=0;
    for (i=bid;i<MAX_BUCKET_SIZE;i+=num_block){
        cpu::vertex *temp_v = &gpu_global_vertex[gpu_vertex_buf[i]];
        int num_edges = gpu_global_vertex[gpu_vertex_buf[i]+1].edge_index - temp_v->edge_index;
        int tent_current = temp_v->dist;
        if(gpu_vertex_buf[i] == 0)
            return;
	//else if(gpu_vertex_buf[i]==1354)
	
	//printf("@@@ %d block: %d thread: %d\n",num_edges,num_block,num_thread);

#ifdef DEBUG
        printf("GPU: Vertex: %d, Edges num: %d, Tent Current: %d\n", gpu_vertex_buf[i], num_edges,
                tent_current);
#endif        
        for(j=tid_in_block;j<MAX_RESULT_SIZE;j+=num_thread){
            int dist_current = 0;
            int dest = 0;
            int tent_dest = 0;
            int flag = 0;
            if(j < num_edges){
                dist_current = gpu_global_edge[temp_v->edge_index+j].distance;
                dest = gpu_global_edge[temp_v->edge_index+j].des_v;
                tent_dest = gpu_global_vertex[dest].dist;
            }
            if(tent_current + dist_current < tent_dest){
                gpu_global_vertex[dest].dist = tent_current + dist_current;
                flag =1;
            }
            //if(flag)
            //printf("GPU: %d : %d, %d, %d. flag: %d\n", i, dest, tent_dest,
            //        (tent_current + dist_current), flag);   

	    //FIXME
		gpu_used_result_buf[j+32*bid].index = dest*flag;
            	gpu_used_result_buf[j+32*bid].old_distance = tent_dest*flag;
            	gpu_used_result_buf[j+32*bid].new_distance = (tent_current+dist_current)*flag;
if(dest*flag==1275){
	printf("@@@%d %d\n",j+32*bid,gpu_used_result_buf[j+32*bid].index);
	gpu_used_result_buf[j+32*bid].index = dest*flag;
	}

        }
        //dprintf("%d\n", gpu_global_vertex[4].dist);
        //sync for sent result back
        //FIXME
        //__syncthreads();
       //cudaMemcpy((cpu_result+blockDim.x * blockIdx.x),gpu_used_result_buf, sizeof(cpu::gpuResult)*num_thread, cudaMemcpyDeviceToHost);

       //cutilSafeCall(cudaMemcpy((cpu_result+blockDim.x * blockIdx.x),gpu_result_buf, sizeof(cpu::gpuResult)*num_thread, cudaMemcpyDeviceToHost));
    }
    }
    //cutilSafeCall(cudaFree(gpu_result_buf));  

#endif
