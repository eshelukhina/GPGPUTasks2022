#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 16

__kernel void matrix_multiplication1(__global const float* a, __global float* b,
                                    __global float* c, unsigned int M, unsigned int K, unsigned int N) {
    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    __local float buffer_a[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    __local float buffer_b[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    if (global_x < M && global_y < K) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < K / WORK_GROUP_SIZE; t++) {

            float a_temp = 0.0f;
            if (local_x < K) {
                a_temp = a[global_y * K + (t * WORK_GROUP_SIZE + local_x)];
            }
            buffer_a[local_y * WORK_GROUP_SIZE + local_x] = a_temp;

            float b_temp = 0.0f;
            if (local_y < K) {
                b_temp = b[(t * WORK_GROUP_SIZE + local_y) * N + global_x];
            }
            buffer_b[local_y * WORK_GROUP_SIZE + local_x] = b_temp;

            barrier(CLK_LOCAL_MEM_FENCE);
            for (unsigned int k = 0; k < WORK_GROUP_SIZE; k++) {
                sum += buffer_a[local_y * WORK_GROUP_SIZE + k] * buffer_b[k * WORK_GROUP_SIZE + local_x];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        c[global_y * N + global_x] = sum;
    } else {
        return;
    }
}


#define WPT 4

__kernel void matrix_multiplication2(__global const float* a, __global float* b,
                                    __global float* c, unsigned int M, unsigned int K, unsigned int N) {
    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    __local float buffer_a[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    __local float buffer_b[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    float sum[WPT];
    int RTS = WORK_GROUP_SIZE / WPT;

    if (global_x < M && global_y < K) {
        for (unsigned int i = 0; i < WPT; i++) {
            sum[i] = 0.0f;
        }
        for (unsigned int t = 0; t < K / WORK_GROUP_SIZE; t++) {
            for (unsigned int i = 0; i < WPT; i++) {
                float a_temp = 0.0f;
                if (local_x < K) {
                    a_temp = a[global_y * K + (t * WORK_GROUP_SIZE + local_x) + i * RTS];
                }
                buffer_a[local_y * WORK_GROUP_SIZE + local_x + i * RTS] = a_temp;

                float b_temp = 0.0f;
                if (local_y < K) {
                    b_temp = b[(t * WORK_GROUP_SIZE + local_y) * N + global_x + i * RTS];
                }
                buffer_b[local_y * WORK_GROUP_SIZE + local_x + i * RTS] = b_temp;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            for (unsigned int k = 0; k < WORK_GROUP_SIZE; k++) {
                for (unsigned int i = 0; i < WPT; i++) {
                    sum[i] += buffer_a[local_y * WORK_GROUP_SIZE + k] * buffer_b[k * WORK_GROUP_SIZE + local_x + i * RTS];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (unsigned int i = 0; i < WPT; i++) {
            c[global_y * N + global_x + i * RTS] = sum[i];
        }
    } else {
        return;
    }
}