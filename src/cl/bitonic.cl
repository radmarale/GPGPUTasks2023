#define WORKGROUP_SIZE 256

void sort(__global float *a, __global float *b, int parity) {
    if (parity == 0 && *a > *b || parity == 1 && *a < *b) {
        float p = *a;
        *a = *b;
        *b = p;
    }
}

__kernel void bitonic(__global float *as, int n, int step, int loglength) {
    int gid = get_global_id(0);
    int as_index = gid + (gid >> loglength << loglength);
    int sequence_parity = (as_index >> step) & 1;
    sort(as + as_index, as + as_index + (1 << loglength), sequence_parity);
}
