

kernel void VectorAdd(global const float* a, global const float* b, global float* c, int numElements) {
    int iGID = get_global_id(0);

    if (iGID >= numElements)
        return;

    c[iGID] = a[iGID] + b[iGID];
}

kernel void MultiplyMatrices(global const float* a, global const float* b, global float* c) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    int dimension = get_global_size(0);

    float sum = 0.0;
    for(int i = 0; i < dimension; ++i)
        sum += a[row * dimension + i] * b[i * dimension + col];

    c[row * dimension + col] = sum;
}

kernel void MultiplyMatricesShared(global const float* a, global const float* b, global float* c, local float* aShared, local float* bShared) {
    int rowGlobal = get_global_id(0);
    int colGlobal = get_global_id(1);

    int workSize = get_local_size(0);
    int dimension = get_global_size(0);

    int rowLocal = get_local_id(0);
    int colLocal = get_local_id(1);

    float sum = 0.0;
    for(int i = 0; i < get_num_groups(0); ++i) {
        aShared[rowLocal * workSize + colLocal] = a[rowGlobal * dimension + i * workSize + colLocal];
        bShared[rowLocal * workSize + colLocal] = b[(i * workSize + rowLocal) * dimension + colGlobal];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < dimension; ++j)
            sum += aShared[rowLocal * workSize + j] * bShared[j * workSize + colLocal];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[rowGlobal * dimension + colGlobal] = sum;
}