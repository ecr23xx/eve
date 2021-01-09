#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <utility>

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void nnSearch(const int nthreads, const int M, const T *query,
                         const T *ref, long *idx, float *dist) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float minDist = INFINITY;
    int minIdx = -1;

    float queryX = query[index * 3 + 0];
    float queryY = query[index * 3 + 1];
    float queryZ = query[index * 3 + 2];

    float refX, refY, refZ, tempDist;

    for (int j = 0; j < M; j++) {
      refX = ref[j * 3 + 0];
      refY = ref[j * 3 + 1];
      refZ = ref[j * 3 + 2];

      tempDist = (queryX - refX) * (queryX - refX) +
                 (queryY - refY) * (queryY - refY) +
                 (queryZ - refZ) * (queryZ - refZ);
      if (tempDist < minDist) {
        minDist = tempDist;
        minIdx = j;
      }
    } // forj

    idx[index] = minIdx;
    dist[index] = minDist;
  }
}

std::pair<at::Tensor, at::Tensor> nnSearch_cuda(const at::Tensor &query,
                                                const at::Tensor &ref) {
  AT_ASSERTM(query.device().is_cuda(),
             "query point cloud must be a CUDA tensor");
  AT_ASSERTM(ref.device().is_cuda(), "ref point cloud must be a CUDA tensor");
  at::TensorArg query_t{query, "query", 1}, ref_t{ref, "ref", 2};

  at::CheckedFrom c = "nnSearch_cuda"; // function name for check
  at::checkAllSameGPU(c, {query_t, ref_t});
  at::checkAllSameType(c, {query_t, ref_t});
  at::cuda::CUDAGuard device_guard(query.device());

  auto N = query.size(0);
  auto M = ref.size(0);

  auto dist = at::empty({N}, query.options());
  auto idx = at::empty({N}, query.options().dtype(at::kLong));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::cuda::ATenCeilDiv(N, 512L), 4096L));
  dim3 block(512);

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "nnSearch", [&] {
    nnSearch<scalar_t><<<grid, block, 0, stream>>>(
        N, M, query.contiguous().data_ptr<scalar_t>(),
        ref.contiguous().data_ptr<scalar_t>(),
        idx.contiguous().data_ptr<long>(), dist.contiguous().data_ptr<float>());
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_pair(idx, dist);
}
