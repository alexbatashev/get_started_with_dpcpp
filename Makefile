ROCM_TARGET = gfx906

omp: 01_image_blur_openmp.cpp
	clang++ -std=c++20 -fopenmp 01_image_blur_openmp.cpp -o omp

spv: 02_image_blur_usm.cpp 03_image_blur_buffers.cpp
	clang++ -fsycl -fsycl-unnamed-lambda 02_image_blur_usm.cpp -o usm_spv
	clang++ -fsycl -fsycl-unnamed-lambda 03_image_blur_buffers.cpp -o buffers_spv

cuda: 02_image_blur_usm.cpp 03_image_blur_buffers.cpp
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda 02_image_blur_usm.cpp -o usm_cuda
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda 03_image_blur_buffers.cpp -o buffers_cuda

# We're not quite there yet
#rocm: 02_image_blur_usm.cpp 03_image_blur_buffers.cpp
#	clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=$(ROCM_TARGET) -fno-sycl-libspirv -fsycl-unnamed-lambda 02_image_blur_usm.cpp -o usm_cuda
#	clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=$(ROCM_TARGET) -fno-sycl-libspirv -fsycl-unnamed-lambda 03_image_blur_buffers.cpp -o buffers_cuda
