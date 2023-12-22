__kernel void prefix_sum(__global unsigned int* as, int n, int loglength, int stage) {
    int gid = get_global_id(0);
    if (gid >= n >> loglength) {
        return;
    }
    int as_index = (gid << loglength) + (1 << loglength) - 1;
    if (stage == 0) {
        as[as_index] += as[as_index - (1 << loglength - 1)];
    } else {
        if (as_index + 1 < n) {
            as[as_index + (1 << loglength - 1)] += as[as_index];
        }
    }
}