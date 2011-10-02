//#include "cutil.h"
#include "cutil_inline.h"
//#include "vertex.h"
//#define DEBUG
#include "relax.cu"
#ifdef DEBUG
#define dprintf(x...) printf(x)
#else 
#define dprintf(x...)
#endif

//shortest path function declaration
void cal_shortest_path();

int main(void){
    cal_shortest_path();
    //CUT_EXIT();
}

void gpu_memory_prep(cpu &cpu_instance){
     //malloc in GPU
    CUDA_SAFE_CALL(cudaMalloc((void **)&cpu_instance.gpu_vertex,(cpu_instance.vertex_size+2)*sizeof(cpu::vertex)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cpu_instance.gpu_edge,cpu_instance.edges_size*sizeof(cpu::edge)));

    //copy to GPU
    CUDA_SAFE_CALL(cudaMemcpy(cpu_instance.gpu_vertex,cpu_instance.global_vertex,
			(cpu_instance.vertex_size+2)*sizeof(cpu::vertex),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cpu_instance.gpu_edge,cpu_instance.global_edge,
			cpu_instance.edges_size*sizeof(cpu::edge), cudaMemcpyHostToDevice));

    //alloc 2 buffer in GPU
    int vertex_buf_size = V_BUF_SIZE * sizeof(int);
    CUDA_SAFE_CALL(cudaMalloc((void **)&cpu_instance.gpu_vertex_buf,vertex_buf_size));
    int result_size = MAX_RESULT_SIZE * sizeof(cpu::gpuResult)*NUM_BLOCK;
    CUDA_SAFE_CALL(cudaMalloc((void **)&cpu_instance.gpu_used_result_buf,vertex_buf_size));

    //alloc cpu memory
    cpu_instance.vertex_buf_ptr =(int *) malloc(vertex_buf_size);
    cpu_instance.gpu_result_buf =(cpu::gpuResult *) malloc(result_size);

printf("%llx %llx\n",cpu_instance.vertex_buf_ptr,cpu_instance.gpu_result_buf);
       printf("~~%llx %llx %d\n",cpu_instance.gpu_result_buf,cpu_instance.gpu_used_result_buf,result_size);
/*
    cudaDeviceProp deviceProp;
    cutilSafeCall(cudaGetDeviceProperties(&deviceProp, cutGetMaxGflopsDeviceId()));
    if(!deviceProp.canMapHostMemory){
	printf("Do not support Mapped Memory\n");
	exit(1);
    }
    //alloc mapped memory in cpu
    int vertex_buf_size = V_BUF_SIZE * sizeof(int);
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&cpu_instance.vertex_buf_ptr,vertex_buf_size , cudaHostAllocMapped));
    CUDA_SAFE_CALL(cudaHostGetDevicePointer((void **)&cpu_instance.gpu_vertex_buf, (void *)cpu_instance.vertex_buf_ptr, 0));

    int result_size = MAX_RESULT_SIZE*sizeof(cpu::gpuResult);
    for(int i=0;i<NUM_BLOCK;i++){
         CUDA_SAFE_CALL(cudaHostAlloc((void **)&cpu_instance.gpu_result_buf[i],result_size , cudaHostAllocMapped));
	 CUDA_SAFE_CALL(cudaHostGetDevicePointer((void **)&cpu_instance.gpu_used_result_buf[i],
	 					        (void *)cpu_instance.gpu_result_buf[i], 0));	 
    }
*/
}

void parse_result(cpu &cpu_instance){
     int result_count = 0;

     for(int i=0;i<NUM_BLOCK;i++){
       cpu::gpuResult *current_result = &cpu_instance.gpu_result_buf[i*MAX_RESULT_SIZE];

       while(1){
            if(result_count >= MAX_RESULT_SIZE){
                break;
            }
            if(current_result[result_count].index == 0){
                result_count++;
                break;//continue;
             }

printf("%d %d %d\n",current_result[result_count].index,current_result[result_count].old_distance,current_result[result_count].new_distance);            

            int old_index = current_result[result_count].old_distance / cpu_instance.delta;
            int new_index = current_result[result_count].new_distance / cpu_instance.delta;
            if(current_result[result_count].old_distance != MAX_DISTANCE){
                cpu_instance.bucket_array[old_index].erase(current_result[result_count].index);
	    }
            cpu_instance.bucket_array[new_index].insert(current_result[result_count].index);
            result_count++;
        }
     }
}

void cal_shortest_path(){
    int num_block = NUM_BLOCK;
    int num_threads = 32;
    dim3 dg(num_block, 1, 1);
    dim3 db(num_threads, 1, 1);
    cpu cpu_instance("nys.gr");

    cudaSetDevice(cutGetMaxGflopsDeviceId());

    gpu_memory_prep(cpu_instance);

    int min;
    int vertex_buf_size = V_BUF_SIZE * sizeof(int);
    int result_size = MAX_RESULT_SIZE*sizeof(cpu::gpuResult)*NUM_BLOCK;

   while(!cpu_instance.is_all_bucket_empty()){

        min = cpu_instance.min_no_empty_bucket();

	//set v set to zero
	memset(cpu_instance.vertex_buf_ptr,0,vertex_buf_size);

	//copy&erase vertex in min bucket
        int count = cpu_instance.bucket_set_to_array(min, cpu_instance.vertex_buf_ptr);
        printf("min: %d  count: %d\n", min,count);

	//deploy vertex set to GPU
	CUDA_SAFE_CALL(cudaMemcpy(cpu_instance.gpu_vertex_buf,cpu_instance.vertex_buf_ptr,
				vertex_buf_size,cudaMemcpyHostToDevice));

        //call cuda function
        relax_all<<<num_block,num_threads>>>(cpu_instance.gpu_vertex_buf,cpu_instance.gpu_used_result_buf,
               cpu_instance.gpu_vertex,cpu_instance.gpu_edge);
       CUT_CHECK_ERROR("Kernel execution failed\n");

       //get the result back
       CUDA_SAFE_CALL(cudaThreadSynchronize());
       //printf("~o~\n");
       printf("~~%llx %llx %d\n",cpu_instance.gpu_result_buf,cpu_instance.gpu_used_result_buf,result_size);
       CUDA_SAFE_CALL(cudaMemcpy(cpu_instance.gpu_result_buf,cpu_instance.gpu_used_result_buf,
				result_size,cudaMemcpyDeviceToHost));

       //get the result from gpu
       parse_result(cpu_instance);
       
    }
    get_result<<<1,1>>>(cpu_instance.gpu_vertex,4);
    printf("over\n");
    
}

cpu::cpu(char* filepath){
    init_memory(filepath);
    delta = 0xff;
    src = 1;
    global_vertex[src].dist = 0;
    dest = 4;
    init_all_bucket();
}

cpu::~cpu(){
    free(global_vertex);
    free(global_edge);
    free(gpu_result_buf);
    free(vertex_buf_ptr);
    CUDA_SAFE_CALL(cudaFree(gpu_vertex));
    CUDA_SAFE_CALL(cudaFree(gpu_edge));
    CUDA_SAFE_CALL(cudaFree(gpu_vertex_buf));
    CUDA_SAFE_CALL(cudaFree(gpu_used_result_buf));
/*
    //mapped memory
    CUDA_SAFE_CALL(cudaFreeHost(vertex_buf_ptr));
    for(int i=0;i<NUM_BLOCK;i++)
        CUDA_SAFE_CALL(cudaFreeHost(gpu_result_buf[i]));
*/
}

int cpu::init_graph(){
  int i;
 //vertex size and edge size
  global_vertex =(struct vertex*) malloc((vertex_size+2)*sizeof(struct vertex));
  global_edge = (struct edge*)malloc(edges_size*sizeof(struct edge));

  //init vertex
  for(i=0;i<vertex_size+2;i++){
    global_vertex[i].edge_index =0;
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
	  //global_vertex[src].edges=&global_edge[cur_edge-1];
	  global_vertex[src].edge_index=cur_edge-1;	
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
  dprintf("end of init\n");
  //copy to GPU
 /* CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_vertex,(vertex_size+2)*sizeof(struct vertex)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_edge,edges_size*sizeof(struct edge)));
  CUDA_SAFE_CALL(cudaMemcpy(gpu_vertex,global_vertex,(vertex_size+2)*sizeof(struct vertex),cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(gpu_edge,global_edge,edges_size*sizeof(struct edge)));
  */
  return 0;
}

int cpu::init_all_bucket(){
    dprintf("insert src : %d\n", src);
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

//copy to buffer and erase
int cpu::bucket_set_to_array(int index, int* array){
    int count = 0;
    std::set<int>::iterator it = bucket_array[index].begin();
    for(;it!=bucket_array[index].end();it++){
            array[count]=*it;
	    bucket_array[index].erase(it);
            count++;
//	    if(count>V_BUF_SIZE){
	    if(count>NUM_BLOCK){	
		printf("oops!\n");
		return count;
	    }
    }
    //for(int i=count;i<V_BUF_SIZE;i++)
    //	    array[i]=0;
    return count;
}

