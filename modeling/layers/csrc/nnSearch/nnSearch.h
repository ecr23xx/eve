#pragma once
#include <torch/types.h>

#ifdef WITH_CUDA
/**
 * TODO: batch
 * C interface for point cloud nearest neighbor search
 * @param query query point cloud with size Nx3
 * @param ref reference point cloud with size Nx3
 * @return tensor with size Nx2, where the first column is index, distance and
 * the second column is distance
 */
std::pair<at::Tensor, at::Tensor> nnSearch_cuda(const at::Tensor& src,
                                                const at::Tensor& dst);
#endif

/**
 * TODO: batch
 * C interface for point cloud nearest neighbor search
 * @param query query point cloud with size Nx3
 * @param ref reference point cloud with size Nx3
 * @return tensor with size Nx2, where the first column is index, distance and
 * the second column is distance
 */
std::pair<at::Tensor, at::Tensor> nnSearch_cpu(const at::Tensor& src,
                                               const at::Tensor& dst);

// Python interface
inline std::pair<at::Tensor, at::Tensor> nnSearch(const at::Tensor& src,
                                                  const at::Tensor& dst) {
  if (src.type().is_cuda()) {
#ifdef WITH_CUDA
    return nnSearch_cuda(src, dst);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return nnSearch_cpu(src, dst);
}