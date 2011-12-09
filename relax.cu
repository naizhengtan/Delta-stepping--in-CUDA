#ifndef _RELAX_H_
#define _RELAX_H_
#include "vertex.h"

__global__ void
get_result(cpu::vertex* gpu_global_vertex,int des,int src){
	printf("result: %d\n",gpu_global_vertex[des].dist);	
	int pre = gpu_global_vertex[des].pre_vertex;
	printf("(%d,%d)",des,gpu_global_vertex[des].dist);
	while(pre != src){
	   printf(" <- (%d,%d)",pre,gpu_global_vertex[pre].dist);
	   pre = gpu_global_vertex[pre].pre_vertex;
	}
	printf(" <- (%d,%d)\n",src,gpu_global_vertex[src].dist);
}

__global__ void
verify_result(cpu::vertex* gpu_global_vertex,cpu::gpuResult *gpu_result){

const unsigned int tid = threadIdx.x;

//    for(int i=0;i<NUM_BLOCK;i++){
       int count=0;
       cpu::gpuResult* current = &gpu_result[tid*MAX_RESULT_SIZE];
       while(1){
       if(current[count].index==0)
          break;
       	  if(current[count].new_distance<gpu_global_vertex[current[count].index].dist){
//	  	printf("!!!!\n");
		gpu_global_vertex[current[count].index].dist = current[count].new_distance;
		gpu_global_vertex[current[count].index].pre_vertex = current[count].pre;
       	  }
       count++;
       }
//    }
}

__global__ void
insert_src(cpu::gpuSet *gpu_used_set, int src){
    gpu_used_set[0].index[src] = 1;
    gpu_used_set[0].index[SET_MAX_ELEMENT - 1] = 1;
    printf("INSERT\n");
}

__global__ void
bucket_ops(cpu::gpuResult *gpu_result, cpu::gpuSet *gpu_used_set, int delta){
    printf("hello world\n");
    const unsigned int tid = threadIdx.x;
    int count = 0;
    cpu::gpuResult* current = &gpu_result[tid*MAX_RESULT_SIZE];
    while(1){
        if(current[count].index==0)
            break;
        gpu_used_set[current[count].old_distance/delta].index[SET_MAX_ELEMENT - 1] -=
            gpu_used_set[current[count].old_distance/delta].index[current[count].index];
        gpu_used_set[current[count].new_distance/delta].index[SET_MAX_ELEMENT - 1] +=
            (1 - gpu_used_set[current[count].new_distance/delta].index[current[count].index]);
        gpu_used_set[current[count].old_distance/delta].index[current[count].index]
            = 0;
        gpu_used_set[current[count].new_distance/delta].index[current[count].index]
            = 1;
        current[count].index = 0;
        count++;
    }
}

__global__ void min_bucket(int* gpu_vertex_buf, cpu::gpuSet *gpu_used_set, int*
        d_return_v){
    int i = 0;
    for(i = 0; i < MAX_BUKET_NUM; i++){
        if(gpu_used_set[i].index[SET_MAX_ELEMENT - 1] != 0)
            break;
    }
    int j = 0, k = 0;
    for(k = 0; k < V_BUF_SIZE; k++)
        gpu_vertex_buf[k] = 0;
    k = 0;
    for(j = 0; j < SET_MAX_ELEMENT; j++){
        if(gpu_used_set[i].index[j] == 1){
            gpu_vertex_buf[k] = j;
            k++;
        }
    }
    *d_return_v = i;
    printf("%d, %d\n", i, *d_return_v);
}

__global__ void
relax_all(int* gpu_vertex_buf, cpu::gpuResult* gpu_used_result_buf,
	       cpu::vertex* gpu_global_vertex, cpu::edge* gpu_global_edge){

    const unsigned int bid = blockIdx.x; 
    const unsigned int num_block = gridDim.x; 
    const unsigned int tid_in_block = threadIdx.x;
    const unsigned int num_thread = blockDim.x;
    const unsigned int tid_in_grid = blockDim.x * blockIdx.x +threadIdx.x;

    int i=0,j=0;
    int dist_current,dest,tent_dest;
    __shared__ int result_count,lock;
    if(tid_in_block==0){
	result_count=0;
	lock=0;
    }

    //one vertex per block
    for (i=bid;i<V_BUF_SIZE;i+=num_block){

        if(gpu_vertex_buf[i] == 0)
            return;

	//get current vertex's info
        //cpu::vertex *temp_v = &gpu_global_vertex[gpu_vertex_buf[i]];
	int edge_index = gpu_global_vertex[gpu_vertex_buf[i]].edge_index;
	cpu::gpuResult *current_result_buf = &gpu_used_result_buf[bid*MAX_RESULT_SIZE]; //the buffer now used
        int num_edges = gpu_global_vertex[gpu_vertex_buf[i]+1].edge_index - edge_index;
        int tent_current = gpu_global_vertex[gpu_vertex_buf[i]].dist;

	//one edge per thread
        for(j=tid_in_block;j<num_edges;j+=num_thread){
		//get edge's info
                dist_current = gpu_global_edge[edge_index+j].distance;
                dest = gpu_global_edge[edge_index+j].des_v;
                tent_dest = gpu_global_vertex[dest].dist;

            //if(tent_current + dist_current > MAX_DISTANCE)
                //printf("DISTANCE BOOM\n");

            if(tent_current + dist_current < gpu_global_vertex[dest].dist){
                gpu_global_vertex[dest].dist = tent_current + dist_current;
		gpu_global_vertex[dest].pre_vertex = gpu_vertex_buf[i];
                  
	    //FIXME: bad critical section
	    int now,loop=0;

while(loop==0){
if(atomicExch(&lock,1)==0){
	    now = result_count;
	    atomicAdd(&result_count,1);
	    loop=1;
	    atomicExch(&lock,0);
	    }
}
	current_result_buf[now].index = dest;
    current_result_buf[now].old_distance = tent_dest;
   	current_result_buf[now].new_distance = (tent_current+dist_current);
	current_result_buf[now].pre = gpu_vertex_buf[i];
//if(result_count>=MAX_RESULT_SIZE)
//	printf("OVERFLOW!!!!%d\n", result_count);
//printf("%d %d %d\n",dest,tent_dest,tent_current+dist_current);
//printf("GPU:%d->%d old:%d new:%d %d %d\n",gpu_vertex_buf[i],current_result_buf[now].index,current_result_buf[now].old_distance,current_result_buf[now].new_distance,now,result_count);
        }
	}
    }
 }


#endif
