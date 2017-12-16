

kernel void init(global float* x) {
    x[get_global_id(0)] = 0;
}

kernel void jacobiStep(global const float* a, global const float* b, global float* xNew, global const float* xOld, int dimension) {
    int row = get_global_id(0) / dimension;
    int N = dimension;

    float sum = 0.0f;
    for(int col = 0; col < N; ++col) {
        if(col != row) {
            sum += a[row * N + col] * xOld[col];
        }
    }

    xNew[row] = (b[row] - sum) / a[row * N + row];
}

kernel void difference(global const float* xOld, global const float* xNew, global float* diff, local float* partialSum) {
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

kernel void initRHS(global const float* y, global float* rhs, float h) {
    int i = get_global_id(0);

    if(i == 0 || i == get_global_size(0) - 1) {
        rhs[i] = 0.0f;
    } else {
        rhs[i] = (6.0f / (h * h)) * (y[i + 1] - 2 * y[i] + y[i - 1]);
    }
}

kernel void jacobiSplineStep(global const float* rhs, global const float* cOld, global float* cNew) {
    int i = get_global_id(0);

    if(i > 0) {
        cNew[i] = (rhs[i] - cOld[i - 1] - cOld[i + 1]) / 4.0f;
    }
}

kernel void computeAB(global const float* y, global const float* c, global float* a, global float* b, float h) {
    int i = get_global_id(0);

    if(i > 0) {
        float bi = ((1.0f / h) * (y[i] - y[i - 1])) - ((h / 6.0f) * (c[i] - c[i - 1]));
        b[i] = bi;
        a[i] = y[i - 1] + (0.5f * bi * h) - (0.1667f * c[i - 1] * h * h);
    }
}

kernel void interpolate(global const float* a, global const float* b, global const float* c, global float* y, float h) {
    float x = (float) get_global_id(0);
    int i = (int) (x / h) + 1;
    float lowerBound = (i - 1) * h;
    float upperBound = lowerBound + h;

    y[get_global_id(0)] = (1.0f / (6.0f * h)) * c[i] * ((x - lowerBound) * (x - lowerBound) * (x - lowerBound)) +
           (1.0f / (6.0f * h)) * c[i - 1] * ((upperBound - x) * (upperBound - x) * (upperBound - x)) +
           	b[i] * (x - 0.5f * (lowerBound + upperBound)) + a[i];
}