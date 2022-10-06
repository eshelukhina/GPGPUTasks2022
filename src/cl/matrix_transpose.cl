#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 16

__kernel void matrix_transpose(__global const float* a, __global float* aT, unsigned int M, unsigned int K) {
    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    __local float buffer[(WORK_GROUP_SIZE + 1) * WORK_GROUP_SIZE];
    if (global_x < M && global_y < K) {
        buffer[local_y * (WORK_GROUP_SIZE + 1) + local_x] = a[global_y * K + global_x];
    } else {
        return;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int x = get_group_id(1) * WORK_GROUP_SIZE + local_x;
    unsigned int y = get_group_id(0) * WORK_GROUP_SIZE + local_y;

    if (x < M && y < K) {
        aT[y * M + x] = buffer[local_x * (WORK_GROUP_SIZE + 1) + local_y];
    }
}