//#include "cutil.h"
#include "cutil_inline.h"
//#include "vertex.h"
#include "relax.cu"
#define DEBUG
#ifdef DEBUG
#define dprintf(x...) printf(x)
#else 
#define dprintf(x...)
#endif

int main(void){
    int num_block = 8;
    int num_threads = 32;
    dim3 dg(num_block, 1, 1);
    dim3 db(num_threads, 1, 1);
    cpu cpu_instance("hi.gr");
    cpu::gpuResult *gpu_used_result_buf;

    cudaSetDevice(cutGetMaxGflopsDeviceId());

     //copy to GPU
    CUDA_SAFE_CALL(cudaMalloc((void **)&cpu_instance.gpu_vertex,(cpu_instance.vertex_size+2)*sizeof(cpu::vertex)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cpu_instance.gpu_edge,cpu_instance.edges_size*sizeof(cpu::edge)));
    //malloc danteng!!
    CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_used_result_buf,sizeof(cpu::gpuResult)*num_block*num_threads));  
    //copy  
    CUDA_SAFE_CALL(cudaMemcpy(cpu_instance.gpu_vertex,cpu_instance.global_vertex,(cpu_instance.vertex_size+2)*sizeof(cpu::vertex),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cpu_instance.gpu_edge,cpu_instance.global_edge,cpu_instance.edges_size*sizeof(cpu::edge), cudaMemcpyHostToDevice));

    //malloc vertex buffer
    CUDA_SAFE_CALL(cudaMalloc((void**)&cpu_instance.vertex_buf_ptr, MAX_BUCKET_SIZE));
    //malloc result buffer

    int min;
    int temp_vertex_array[MAX_BUCKET_SIZE];
    
    while(!cpu_instance.is_all_bucket_empty()){
        min = cpu_instance.min_no_empty_bucket(); 
        int count = cpu_instance.bucket_set_to_array(min, temp_vertex_array);
        //FIXME, send count may have better performance
        for(int i=count; i< MAX_BUCKET_SIZE; i++)
                temp_vertex_array[i] = -1;
        //Copy to CPU-GPU buffer
        CUDA_SAFE_CALL(cudaMemcpy(cpu_instance.vertex_buf_ptr,temp_vertex_array, MAX_BUCKET_SIZE,cudaMemcpyHostToDevice));
       //call cuda function
       relax_all<<<dg,db>>>(cpu_instance.vertex_buf_ptr, cpu_instance.gpu_vertex,cpu_instance.gpu_edge,cpu_instance.gpu_result_buf,gpu_used_result_buf);
       //relax_all<<<dg,db>>>(cpu_instance.vertex_buf_ptr, cpu_instance.gpu_vertex,cpu_instance.gpu_edge,cpu_instance.gpu_result_buf);
       CUDA_SAFE_CALL(cudaMemcpy(cpu_instance.gpu_result_buf,gpu_used_result_buf, sizeof(cpu::gpuResult)*num_threads*num_block, cudaMemcpyDeviceToHost)); 
       //get the result from gpu
       int result_count = 0;
       while(1){
            if(result_count >= MAX_BUCKET_SIZE){
                break;
            }
            if(cpu_instance.gpu_result_buf[result_count].index == 0){
                result_count++;
                continue;
             }
            
            int old_index = cpu_instance.gpu_result_buf[result_count].old_distance / cpu_instance.delta;
            int new_index = cpu_instance.gpu_result_buf[result_count].new_distance / cpu_instance.delta;
            cpu_instance.bucket_array[old_index].erase(cpu_instance.gpu_result_buf[result_count].index);
            cpu_instance.bucket_array[new_index].insert(cpu_instance.gpu_result_buf[result_count].index);
            result_count++;
        }
       
    }


    CUDA_SAFE_CALL(cudaFree(cpu_instance.gpu_vertex));
    CUDA_SAFE_CALL(cudaFree(cpu_instance.gpu_edge));
    cutilSafeCall(cudaFree(gpu_used_result_buf));  
    free(cpu_instance.global_vertex);
    free(cpu_instance.global_edge);
    
}

cpu::cpu(char* filepath){
    init_memory(filepath);
    init_all_bucket();
}

cpu::~cpu(){
    free(global_vertex);
    free(global_edge);
}

int cpu::init_graph(){
  int i;
 //vertex size and edge size
  global_vertex =(struct vertex*) malloc((vertex_size+2)*sizeof(struct vertex));
  global_edge = (struct edge*)malloc(edges_size*sizeof(struct edge));

  //init vertex
  for(i=0;i<vertex_size+2;i++){
    global_vertex[i].edges =0;
    global_vertex[i].dist = MAX_DISTANCE;
    global_vertex[i].pre_vertex = -1;
  }

  graph_init=1;
  return 0;
}

int cpu::init_memory(char* filepath){
  char string[256];

  FILE* fp = fopen(filepath,"r");
  if(fp==NULL)
    return -1;

  //init edge, file staff
  while(fgets(string,256,fp)!=NULL){
    static char sign;
    //assumption1: the node start at index 1
    static int src,dest,dist,cur_v=0,cur_edge=0;
    //get the sign of the file line
    sscanf(string,"%c",&sign);

    //the line describe the edge
    if(sign=='a'){
      if(!graph_init)
	return -2;
      if(cur_edge>edges_size)
	return -4;
      sscanf(string,"%c\t%d\t%d\t%d",&sign,&src,&dest,&dist);
      dprintf("edge from:%d to:%d dist:%d\n",src,dest,dist);
      //add the edge to the edge list
      global_edge[cur_edge].des_v=dest;
      global_edge[cur_edge].distance=dist;
      cur_edge++;


      if(cur_v!=src){
	//assumption2: sorted vertex and there is no isolated vertex
	if(cur_v==src-1){
	  global_vertex[src].edges=&global_edge[cur_edge-1];
	  cur_v=src;
	}
	else
	  return -3;
      }
    }
    //the line describe the size of graph
    else if(sign=='p'){ 
      sscanf(string,"%c\tsp\t%d\t%d",&sign,&src,&dest);
      vertex_size = src;
      edges_size = dest;
      #ifdef DEBUG
      dprintf("GOT the size of graph, vertex:%d edge:%d\n",vertex_size,edges_size);
      #endif
      init_graph();
    }
  }

  fclose(fp);
  //copy to GPU
 /* CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_vertex,(vertex_size+2)*sizeof(struct vertex)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_edge,edges_size*sizeof(struct edge)));
  CUDA_SAFE_CALL(cudaMemcpy(gpu_vertex,global_vertex,(vertex_size+2)*sizeof(struct vertex),cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(gpu_edge,global_edge,edges_size*sizeof(struct edge)));
  */
  return 0;
}

int cpu::init_all_bucket(){
    bucket_array[0].insert(src);
}

int cpu::is_all_bucket_empty(){
    return min_no_empty_bucket()==-1;
}

int cpu::min_no_empty_bucket(){
    for(int i=0;i<MAX_BUKET_NUM;i++){
        if(!bucket_array[i].empty()){
              return i;
        }
    }
    return -1;
}

int cpu::bucket_set_to_array(int index, int* array){
    int count = 0;
    std::set<int>::iterator it = bucket_array[index].begin();
    for(;it!=bucket_array[index].end();it++){
            array[count]=*it;
            count++;
        }
    return count;
}

