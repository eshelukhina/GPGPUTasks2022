#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global const float* as, __global float* out, const unsigned int size, const unsigned int n) {
    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    unsigned int a_left = global_id / (n * 2) * (n * 2);
    unsigned int a_right = min(a_left + n, size);
    unsigned int b_left = a_right;
    unsigned int b_right = min(b_left + n, size);

    unsigned int ind = global_id - a_left;
    int left = max((int)ind - (int)(b_right - b_left), 0) - 1;
    int right = min(n, ind);

    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (as[a_left + mid] <= as[b_left + ind - mid - 1]) {
            left = mid;
        } else {
            right = mid;
        }
    }

    unsigned int a = a_left + right;
    unsigned int b = b_left + ind - right;

    if (a < a_right && (b >= b_right || as[a] <= as[b])) {
        out[global_id] = as[a];
    } else {
        out[global_id] = as[b];
    }
}
