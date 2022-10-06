#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUE_PER_WORK_ITEM 32
#define WORK_GROUP_SIZE 128

__kernel void sum_baseline(__global const unsigned int* as, __global unsigned int* res, unsigned int n) {
    int global_id = get_global_id(0);
    if (global_id < n) {
        atomic_add(res, as[global_id]);
    }
}

__kernel void sum_cycle(__global const unsigned int* as, __global unsigned int* res, unsigned int n) {
    int global_id = get_global_id(0);
    unsigned int sum = 0;
    for (int i = 0; i < VALUE_PER_WORK_ITEM; i++) {
        int ind = global_id * VALUE_PER_WORK_ITEM + i;
        if (ind >= n) {
           break;
        }
        sum += as[ind];
    }
    atomic_add(res, sum);
}

__kernel void sum_cycle_coalesced(__global const unsigned int* as, __global unsigned int* res, unsigned int n) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);
    unsigned int sum = 0;
    for (int i = 0; i < VALUE_PER_WORK_ITEM; ++i) {
        int ind = group_id * local_size * VALUE_PER_WORK_ITEM + i * local_size + local_id;
        if (ind >= n) {
            break;
        }
        sum += as[ind];
    }
    atomic_add(res, sum);
}

__kernel void sum_local_memory(__global const unsigned int* as, __global unsigned int* res, unsigned int n) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    unsigned int sum = 0;
    __local int local_as[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_as[local_id] = as[global_id];
    } else {
        local_as[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_as[i];
        }
        atomic_add(res, sum);
    }
}

__kernel void sum_tree(__global const unsigned int* as, __global unsigned int* res, unsigned int n) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    unsigned int sum = 0;
    __local int local_as[WORK_GROUP_SIZE];

    if (global_id < n) {
        local_as[local_id] = as[global_id];
    } else {
        local_as[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = WORK_GROUP_SIZE; i > 1; i/=2) {
        if (2 * local_id < i) {
            const int left = local_as[local_id];
            const int right = local_as[local_id + i/2];
            local_as[local_id] = left + right;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(res, local_as[0]);
    }
}