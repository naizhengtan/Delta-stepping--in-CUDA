#define __vertex_h
#ifdef __vertex_h
#include "stdio.h"
#include "stdlib.h"
#include <set>
#include <sys/time.h>

#define MAX_BUKET_NUM 0x7fff
#define MAX_DISTANCE 0x7fffff
#define MAX_BUCKET_SIZE 2048
#define MAX_RESULT_SIZE 1024
#define NUM_BLOCK 8
#define V_BUF_SIZE 1024

class cpu{
public:

    struct edge{
        int des_v;
        int distance;
    };
     
    struct vertex{
      //struct edge *edges;
        int edge_index;
        int dist;
        int pre_vertex;
    };


    struct gpuResult{
      int index;
      int pre;
      int old_distance;
      int new_distance;
    };


    typedef std::set<int> bucket;

    cpu(char*);
    ~cpu();
    int init_all_bucket();//OK
    int is_all_bucket_empty();
    int min_no_empty_bucket();
    int init_memory(char* filepath);
    int init_graph();

    void print_bucket();

    int bucket_set_to_array(int index, int* array);
    //pure cpu parameters
   int vertex_size,edges_size;
   int graph_init;
   int delta;
   bucket bucket_array[MAX_BUKET_NUM];
   int src ,dest;
   //gpu related
    struct vertex *global_vertex;
    struct edge *global_edge;
    //struct gpuResult *gpu_result_buf[NUM_BLOCK];
    struct gpuResult *gpu_result_buf;
    int* vertex_buf_ptr;
    //gpu used
    struct vertex *gpu_vertex;
    struct edge *gpu_edge;
    int* gpu_vertex_buf; //mapped memory
    //struct gpuResult *gpu_used_result_buf[NUM_BLOCK];//mapped memory
    struct gpuResult *gpu_used_result_buf;
    //profile
    struct timeval start,end;
  
};
#endif
