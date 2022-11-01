#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void local_bitonic(__global float *as, unsigned int n, unsigned int i, unsigned int j) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);

    __local float local_as[WORK_GROUP_SIZE];

    local_as[local_id] = global_id < n ? as[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (; j > 0; j /= 2) {
        if (j + local_id < n && local_id % (2 * j) < j) {
            if (global_id % (2 * i) < i == local_as[local_id] > local_as[local_id + j]) {
                float tmp = local_as[local_id];
                local_as[local_id] = local_as[local_id + j];
                local_as[local_id + j] = tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_id < n) {
        as[global_id] = local_as[local_id];
    }
}


__kernel void bitonic(__global float *as, unsigned int n, unsigned int i, unsigned int j) {
    unsigned int global_id = get_global_id(0);

    if (j + global_id < n && global_id % (2 * j) < j) {
        if (global_id % (2 * i) < i == as[global_id] > as[global_id + j]) {
            float tmp = as[global_id];
            as[global_id] = as[global_id + j];
            as[global_id + j] = tmp;
        }
    }
}
