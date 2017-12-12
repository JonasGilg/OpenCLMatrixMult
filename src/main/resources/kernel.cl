

kernel void init(global float* x) {
    x[get_global_id(0)] = 0;
}

kernel void jacobiStep(global float* a, global float* b, global float* xNew, global float* xOld, int dimension) {
    int row = get_global_id(0) / dimension;
    int N = dimension;

    float sum = 0.0;
    for(int col = 0; col < N; ++col) {
        if(col != row) {
            sum = a[row * N + col] * xOld[col];
        }
    }

    xNew[row] = (b[row] - sum) / a[row * N + row];
}

kernel void difference(global float* xOld, global float* xNew, global float* diff, local float* partialSum) {
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    int localSize = get_local_size(0);

    partialSum[localId] = fabs(xOld[globalId] - xNew[globalId]);

    for(int stride = localSize >> 1; stride >= 32; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(localId < stride)
            partialSum[localId] += partialSum[localId + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(localId < 32) {
        partialSum[localId] += partialSum[localId + 16];
        partialSum[localId] += partialSum[localId + 8];
        partialSum[localId] += partialSum[localId + 4];
        partialSum[localId] += partialSum[localId + 2];
        partialSum[localId] += partialSum[localId + 1];
    }

    diff[get_group_id(0)] = partialSum[0];
}