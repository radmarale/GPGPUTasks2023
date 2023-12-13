#define WORKGROUP_SIZE 256

void sort(__local float *a, __local float *b, int parity) {
    if (parity == 0 && *a > *b || parity == 1 && *a < *b) {
        float p = *a;
        *a = *b;
        *b = p;
    }
}

__kernel void bitonic(__global float *as, int n, int step, int loglength) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int as_index = gid + (gid >> loglength << loglength);
    __local float left_elements_buffer[WORKGROUP_SIZE], right_elements_buffer[WORKGROUP_SIZE];
    left_elements_buffer[lid] = as[as_index];
    right_elements_buffer[lid] = as[as_index + (1 << loglength)];    
    int sequence_parity = (as_index >> step) & 1;
    sort(left_elements_buffer + lid, right_elements_buffer + lid, sequence_parity);
    as[as_index] = left_elements_buffer[lid];
    as[as_index + (1 << loglength)] = right_elements_buffer[lid];
}
