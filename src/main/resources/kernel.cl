

kernel void init(global float* x) {
    x[get_global_id(0)] = 0;
}

kernel void jacobiStep(global float* a, global float* b, global float* xNew, global float* xOld) {
    int i = get_global_id(0);
    int N = get_global_size(0);

    float sum = 0.0;
    for(int k = 0; k < N; ++k) {
        if(k != i) {
            sum = a[i * N + k] * xOld[k];
        }
    }

    xNew[i] = (b[i] - sum) / a[i * N + i];
}

kernel void difference(global float* xOld, global float* xNew, global float* diff, local float* partialSum) {
    int i = get_global_id(0);
    int block = get_local_size(0);

    partialSum[i] = fabs(xOld[i] - xNew[i]);

    for(int stride = block >> 1; stride >= 32; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(block < stride)
            partialSum[i] += partialSum[i + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(i < 32) {
        partialSum[block] += partialSum[block + 16];
        partialSum[block] += partialSum[block + 8];
        partialSum[block] += partialSum[block + 4];
        partialSum[block] += partialSum[block + 2];
        partialSum[block] += partialSum[block + 1];
    }

    diff[block] = partialSum[0];
}

/*
kernel void VectorAdd(global const float* a, global const float* b, global float* c, int numElements) {
    int iGID = get_global_id(0);

    if (iGID >= numElements)
        return;

    c[iGID] = a[iGID] + b[iGID];
}

kernel void MultiplyMatrices(global const float* a, global const float* b, global float* c) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    int dimension = get_global_size(0);

    float sum = 0.0;
    for(int i = 0; i < dimension; ++i)
        sum += a[row * dimension + i] * b[i * dimension + col];

    c[row * dimension + col] = sum;
}

kernel void MultiplyMatricesShared(global const float* a, global const float* b, global float* c, local float* aShared, local float* bShared) {
    int rowGlobal = get_global_id(1);
    int colGlobal = get_global_id(0);

    int workSize = get_local_size(0);
    int dimension = get_global_size(0);
    int gridSize = get_num_groups(0);

    int rowLocal = get_local_id(1);
    int colLocal = get_local_id(0);

    float sum = 0.0;
    for(int i = 0; i < gridSize; ++i) {
        aShared[rowLocal * workSize + colLocal] = a[rowGlobal * dimension + i * workSize + colLocal];
        bShared[rowLocal * workSize + colLocal] = b[(i * workSize + rowLocal) * dimension + colGlobal];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < workSize; ++j)
            sum += aShared[rowLocal * workSize + j] * bShared[j * workSize + colLocal];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[rowGlobal * dimension + colGlobal] = sum;
}*/