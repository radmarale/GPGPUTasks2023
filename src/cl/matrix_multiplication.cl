#define TILE_SIZE 16
#define THREAD_WORK 4

__kernel void matrix_multiplication_global_mem(__global const float* a,
                                               __global const float* b,
                                               __global float* c,
                                               unsigned int M, unsigned int K,
                                               unsigned int N) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);

    if (i < M && j < N) {        
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += a[i * K + k] * b[k * N + j];
        }
        c[i * N + j] = sum;
    }
}

__kernel void matrix_multiplication_local_mem(__global const float* a,
                                              __global const float* b,
                                              __global float* c,
                                              unsigned int M, unsigned int K,
                                              unsigned int N) {
    unsigned int global_i = get_global_id(0);
    unsigned int global_j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0;
    for (int tile_ind = 0; tile_ind * TILE_SIZE < K; ++tile_ind) {
        unsigned int a_i = global_i;
        unsigned int a_j = tile_ind * TILE_SIZE + local_j;
        if (a_i < M && a_j < K) {
            tile_a[local_i][local_j] = a[a_i * K + a_j];
        } else {
            tile_a[local_i][local_j] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int b_i = tile_ind * TILE_SIZE + local_i;
        unsigned int b_j = global_j;
        if (b_i < K && b_j < N) {
            tile_b[local_i][local_j] = b[b_i * N + b_j];
        } else {
            tile_b[local_i][local_j] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[local_i][k] * tile_b[k][local_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_i < M && global_j < N) {
        c[global_i * N + global_j] = sum;
    }
}

__kernel void matrix_multiplication_more_work_per_thread(__global const float* a,
                                                         __global const float* b,
                                                         __global float* c,
                                                         unsigned int M, unsigned int K,
                                                         unsigned int N) {
    unsigned int global_i = get_global_id(0);
    unsigned int global_j = get_global_id(1);

    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);

    unsigned int tile_i = get_group_id(0) * get_local_size(0);
    unsigned int tile_j = get_group_id(1) * get_local_size(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float sum[THREAD_WORK];
    for (int i = 0; i < THREAD_WORK; ++i) {
        sum[i] = 0;
    }

    for (int tile_ind = 0; tile_ind * TILE_SIZE < K; ++tile_ind) {
        for (int w = 0; w < THREAD_WORK; ++w) {
            unsigned int a_i = tile_i + local_i * THREAD_WORK + w;
            unsigned int a_j = tile_ind * TILE_SIZE + local_j;
            if (a_i < M && a_j < K) {
                tile_a[local_i * THREAD_WORK + w][local_j] = a[a_i * K + a_j];
            } else {
                tile_a[local_i * THREAD_WORK + w][local_j] = 0;
            }
        }
        
        for (int w = 0; w < THREAD_WORK; ++w) {
            unsigned int b_i = tile_ind * TILE_SIZE + local_i * THREAD_WORK + w;
            unsigned int b_j = tile_j + local_j;
            if (b_i < K && b_j < N) {
                tile_b[local_i * THREAD_WORK + w][local_j] = b[b_i * N + b_j];
            } else {
                tile_b[local_i * THREAD_WORK + w][local_j] = 0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        

        for (int k = 0; k < TILE_SIZE; ++k) {
            float tmp_b = tile_b[k][local_j];
            for (int w = 0; w < THREAD_WORK; ++w) {
                sum[w] += tile_a[local_i * THREAD_WORK + w][k] * tmp_b;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < THREAD_WORK; ++w) {
        unsigned int c_i = tile_i + local_i * THREAD_WORK + w;
        unsigned int c_j = global_j;
        if (c_i < M && c_j < N) {
            c[c_i * N + c_j] = sum[w];
        }
    }
}