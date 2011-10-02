#ifndef _RELAX_H_
#define _RELAX_H_
#include "vertex.h"

__global__ void
get_result(cpu::vertex* gpu_global_vertex,int i){
	printf("result: %d\n",gpu_global_vertex[i].dist);	
}

__global__ void
relax_all(int* gpu_vertex_buf, cpu::gpuResult* gpu_used_result_buf[],
	       cpu::vertex* gpu_global_vertex, cpu::edge* gpu_global_edge){

    const unsigned int bid = blockIdx.x; 
    const unsigned int num_block = gridDim.x; 
    const unsigned int tid_in_block = threadIdx.x;
    const unsigned int num_thread = blockDim.x;
    const unsigned int tid_in_grid = blockDim.x * blockIdx.x +threadIdx.x;

    int i=0,j=0;
    int dist_current,dest,tent_dest,flag;
    __shared__ int result_count;
    if(tid_in_block==0)
	result_count=0;

    //one vertex per block
    for (i=bid;i<V_BUF_SIZE;i+=num_block){

        if(gpu_vertex_buf[i] == 0)
            return;

	//get current vertex's info
        cpu::vertex *temp_v = &gpu_global_vertex[gpu_vertex_buf[i]];
	cpu::gpuResult *current_result_buf = gpu_used_result_buf[bid]; //the buffer now used
        int num_edges = gpu_global_vertex[gpu_vertex_buf[i]+1].edge_index - temp_v->edge_index;
        int tent_current = temp_v->dist;

	//one edge per thread
        for(j=tid_in_block;j<num_edges;j+=num_thread){

		//get edge's info
                dist_current = gpu_global_edge[temp_v->edge_index+j].distance;
                dest = gpu_global_edge[temp_v->edge_index+j].des_v;
                tent_dest = gpu_global_vertex[dest].dist;

            if(tent_current + dist_current < tent_dest){
                gpu_global_vertex[dest].dist = tent_current + dist_current;
                flag =1;
            }

	    //FIXME
	    atomicAdd(&result_count,1);
		current_result_buf[result_count].index = dest*flag;
            	current_result_buf[result_count].old_distance = tent_dest*flag;
            	current_result_buf[result_count].new_distance = (tent_current+dist_current)*flag;
printf("GPU: %d %d %d\n",dest,tent_dest,tent_current+dist_current);
        }
    }
 }

#endif
