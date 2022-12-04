#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 16

__kernel void matrix_multiplication1(__global const float* a, __global float* b,
                                     __global float* c, unsigned int M, unsigned int K, unsigned int N,
                                     unsigned int UPPER_BOUND_K) {
    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    __local float buffer_a[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    __local float buffer_b[WORK_GROUP_SIZE * WORK_GROUP_SIZE];

    float sum = 0.0f;
    for (unsigned int t = 0; t * WORK_GROUP_SIZE < UPPER_BOUND_K; t++) {
        if (t * WORK_GROUP_SIZE + local_x < K && global_y < M) {
            buffer_a[local_y * WORK_GROUP_SIZE + local_x] = a[global_y * K + (t * WORK_GROUP_SIZE + local_x)];
        }
        if (t * WORK_GROUP_SIZE + local_y < K && global_x < N) {
            buffer_b[local_y * WORK_GROUP_SIZE + local_x] = b[(t * WORK_GROUP_SIZE + local_y) * N + global_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int k = 0; k < WORK_GROUP_SIZE; k++) {
            if (t * WORK_GROUP_SIZE + k < K && global_x < N && global_y < M) {
                sum += buffer_a[local_y * WORK_GROUP_SIZE + k] * buffer_b[k * WORK_GROUP_SIZE + local_x];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_x < N && global_y < M) {
        c[global_y * N + global_x] = sum;
    }
}


#define WPT 4

__kernel void matrix_multiplication2(__global const float* a, __global float* b,
                                     __global float* c, unsigned int M, unsigned int K, unsigned int N,
                                     unsigned int UPPER_BOUND_K) {
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int global_x = get_group_id(0) * WORK_GROUP_SIZE + local_x;
    const unsigned int global_y = get_group_id(1) * WORK_GROUP_SIZE + local_y;

    __local float buffer_a[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    __local float buffer_b[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    float sum[WPT];
    int RTS = WORK_GROUP_SIZE / WPT;

    for (unsigned int i = 0; i < WPT; i++) {
        sum[i] = 0.0f;
    }
    for (unsigned int t = 0; t * WORK_GROUP_SIZE < UPPER_BOUND_K; t++) {
        for (unsigned int i = 0; i < WPT; i++) {
            if (t * WORK_GROUP_SIZE + local_x < K && global_y + i * RTS < M) {
                buffer_a[(local_y + i * RTS) * WORK_GROUP_SIZE + local_x] = a[(global_y + i * RTS) * K + (t * WORK_GROUP_SIZE + local_x)];
            }
            if (t * WORK_GROUP_SIZE + local_y + i * RTS < K && global_x < N) {
                buffer_b[(local_y + i * RTS) * WORK_GROUP_SIZE + local_x] = b[(t * WORK_GROUP_SIZE + local_y + i * RTS) * N + global_x];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int k = 0; k < WORK_GROUP_SIZE; k++) {
            const float tmp = buffer_b[k * WORK_GROUP_SIZE + local_x];
            for (unsigned int i = 0; i < WPT; i++) {
                if (t * WORK_GROUP_SIZE + k < K && global_x < N && global_y + i * RTS < M) {
                    sum[i] += buffer_a[(local_y + i * RTS) * WORK_GROUP_SIZE + k] * tmp;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (unsigned int i = 0; i < WPT; i++) {
        if (global_x < N && global_y + i * RTS < M) {
            c[(global_y + i * RTS) * N + global_x] = sum[i];
        }
    }
}