# Getting Started with Intel oneAPI DPC++

These are training materials for Intel Summer Internship School.

## Step 0. Acquiring Data Parallel C++ compiler

There're three primary sources to get DPC++ compiler:

1) DPC++ is part of oneAPI Base Toolkit: https://software.intel.com/content/www/us/en/develop/tools/oneapi/all-toolkits.html#base-kit
2) DPC++ is an open-source project, that can be built from scratch: https://intel.github.io/llvm-docs/GetStartedGuide.html
3) Intel DevCloud: https://www.intel.com/content/www/us/en/forms/idz/devcloud-enrollment/oneapi-request.html

You also need to clone this repository to be able to interact with the code:

```bash
git clone --recursive https://github.com/alexbatashev/get_started_with_dpcpp.git
```

**NOTE** If you missed the `--recursive` argument when cloning repo do:
```bash
git submodule init
git submodule update
```

## Step 1. The image blur application.

This short guide focuses on porting a simple image blur application from good
old OpenMP to SYCL.

The idea of the algorithm is very simple. It slides over an image with a window
and replaces each pixel with the average value of pixels in the window:

```c++
void blur_image(const Image &in, Image &out) {
  #pragma parallel for collapse(2)
  for (size_t i = 0; i < in.get_width(); i++) {
    for (size_t j = 0; j < in.get_height(); j++) {
      constexpr int radius = 4;

      std::array<unsigned int, 3> sum{0};
      for (int x = -radius; x < radius; x++) {
        for (int y = -radius; y < radius; y++) {
          constexpr size_t zero = 0;
          size_t realX = std::clamp(i + x, zero, in.get_width());
          size_t realY = std::clamp(j + y, zero, in.get_height());

          Pixel p = in(realX, realY);
          sum[0] += p.getR();
          sum[1] += p.getG();
          sum[2] += p.getB();
        }
      }

      constexpr unsigned int area = (2 * radius + 1) * (2 * radius + 1);
      sum[0] /= area;
      sum[1] /= area;
      sum[2] /= area;

      out.setPixel(sum, i, j);
    }
  }
}
```

This is known as [box blur](https://en.wikipedia.org/wiki/Box_blur).

To load image from file we use the [stb_image](https://github.com/nothings/stb/blob/master/stb_image.h)
library. The `Image` class handles both opening existing file and writing to a
new one:

```c++
class Image {
public:
  enum class mode {
    read_only,
    create
  };

  Image(std::filesystem::path path, mode mode = mode::read_only);

  Image(std::filesystem::path path, size_t width, size_t height, mode mode = mode::create);

  size_t get_width() const noexcept;
  size_t get_height() const noexcept;

  const Pixel *get_pointer() const noexcept;
  Pixel *get_pointer() noexcept;

  Pixel operator()(size_t x, size_t y) const noexcept;

  void setPixel(Pixel p, size_t x, size_t y);
};
```

The `Pixel` file is a simple wrapper over three `unsigned char` values, that
handles pixel value clamp for us.

```c++
class Pixel {
public:
  Pixel() = default;

  Pixel(byte r, byte g, byte b);

  Pixel(const std::array<unsigned int, 3> &inp);

  Pixel &operator=(std::array<unsigned int, 3> inp);

  operator std::array<unsigned int, 3>() const noexcept;

  byte getR() const noexcept;
  byte getG() const noexcept;
  byte getB() const noexcept;
};
```

This was a brief overview of what the application looks like. The full code is
provided in [01_image_blur_openmp.cpp](https://github.com/alexbatashev/get_started_with_dpcpp/blob/master/01_image_blur_openmp.cpp).

## Step 2. Unified Shared Memory.

Before we dive deep into the code, let's figure out what is SYCL.

From the [specification](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html):

> SYCL (pronounced “sickle”) is a royalty-free, cross-platform abstraction C++
> programming model for heterogeneous computing. SYCL builds on the underlying
> concepts, portability and efficiency of parallel API or standards like OpenCL
> while adding much of the ease of use and flexibility of single-source C++.

Data Parallel C++ derives from SYCL standard, adding some useful extension on
top of it. So, any valid SYCL application is a valid DPC++ application as well.

In SYCL we have a notion of host and device. A host is a system, that is running
your application. A device is some kind of accelerator, that can be used to
offload compute-demanding tasks to. A device can be the same CPU as your host
system, a GPU, or even an FPGA.

Devices are grouped into platforms. You can see the list of available platforms
by calling `sycl-ls` utility.

All workloads must be submitted to a queue. There are many ways to create a
queue.

First, you need to include the default SYCL header.

```c++
#include <CL/sycl.hpp>
```

**NOTE** SYCL2020 replaces this file with `sycl/sycl.hpp`, but by the time of
writing the closed-source DPC++ compiler doesn't support it.

The easiest way to create a queue would be to simply create an object:

```c++
sycl::queue myQueue;
```

In this case the SYCL runtime will use some heuristics to select the best device
out of the available devices. The following statement would be equivalent:

```c++
sycl::queue myQueue{sycl::default_selector{}};
```

The `default_selector` is the device selection heuristic. There's also a
`cpu_selector`, `gpu_selector`, `accelerator_selector`. Users can also define
custom selectors. See [4.6.1. Device selection](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:device-selection)
for more info.

You can also create a queue from particular devices. See [4.6.5.1. Queue interface](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_queue_interface).

Now that we know how to choose our accelerator, it is time to get our data onto
it. Since we are porting an existing application, a wise choice for us can be
[Unified Shared Memory](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm).

When programming for CPU, you typically have access to your RAM. You can
allocate and deallocate chunks of memory, or do pointer arithmetics. With
accelerators the reality is a bit more complicated. Device memory can be the
same as your host memory when you use the same CPU for host and computation.
GPUs can share RAM with the CPU (like integrated GPUs do), but have a different
interface to work with it, or they may have their own memory hierarchy (like
dGPUs do), with multiple types of memory, so you can't simply allocate a piece
of memory or do pointer arithmetics on it. USM simplifies this for us, providing
a malloc/free-like interface to accelerator memory. There're three kinds of
memory, defined by USM:
1) Device pointers;
2) Shared pointers;
3) Host pointers.

Each USM-capable implementation is required to support device and host pointers.
As the name implies, device memory is allocated on device, and is not accessible
on host. Host memory is allocated on host. Devices, that support host memory
allocation, may access it. Shared allocations are managed by accelerator driver
to provide optimal access for both host and device. In this tutorial we focus on
device allocations only.

Let's have a look at how we can use USM to get our image to the device.

First, let's define a `DeviceImage` class, that will serve a simple wrapper over
a device pointer:

```c++
class DeviceImage {
public:
  DeviceImage(sycl::queue q, size_t width, size_t height) : mWidth(width), mHeight(height) {
    mData = sycl::malloc_device<byte>(width*height*3, q);
  }

  byte *get_pointer() noexcept {
    return mData;
  }

  const byte *get_pointer() const noexcept {
    return mData;
  }

  size_t get_width() const noexcept { return mWidth; }
  size_t get_height() const noexcept { return mHeight; }

  Pixel operator()(size_t x, size_t y) const noexcept {
    byte *bytes = mData + mWidth * x * 3 + y * 3;
    return Pixel(bytes[0], bytes[1], bytes[2]);
  }

  void setPixel(Pixel p, size_t x, size_t y) const {
    byte *bytes = mData + mWidth * x * 3 + y * 3;
    bytes[0] = p.getR();
    bytes[1] = p.getG();
    bytes[2] = p.getB();
  }

private:
  size_t mWidth;
  size_t mHeight;
  byte *mData;
};
```

Now we need to allocate memory for our images and copy the input image to the
device.

```c++
int main() {
  // ...

  DeviceImage inDev{q, inImg.get_width(), inImg.get_height()};
  DeviceImage outDev{q, inImg.get_width(), inImg.get_height()};

  sycl::event cpyEvt = q.memcpy(inDev.get_pointer(), inImg.get_pointer(), 3 * inImg.get_height() * inImg.get_width());
  // ...
}
```

The last line asks SYCL runtime to submit memory copy task to our queue. This
task will be executed asynchronously. In return we get a `sycl::event` object,
that can be used to track completion status of our task. To wait for event
completion, one can call `sycl::event::wait` method, but we will skip it for
now.

Now that the image is on its way to the device memory, let's re-write our code
to be able to run on accelerator devices. We start by changing the signature of
our `blur_image` function to return an event and accept `DeviceImage`s instead.
Next, we submit our actual workload to the queue. To do that, we call the
`sycl::queue::submit` method and feed it with a lambda, accepting `sycl::handler`
argument. This `sycl::handler` class represents what we call Command Group
Handler. It is used to record particular steps needed to execute a task.
The next line tells SYCL runtime, that this task depends on whatever is the
result of the event (in our case memcpy event), and we need to wait for its
completion before this task can be submitted. Now we can replace the two nested
for loops with `sycl::handler::parallel_for` clause, that accepts a range (you
can think of it as a loop boundaries) and a kernel (you can think of it as loop
body). You can find some similarities with TBB's `parallel_for` here. The rest
of the code is pretty much unchanged. I took the liberty to replace some
`std::array` usages here and in `Pixel` class to save a few lines of code. Also,
`std::` math builtins are not available in SYCL kernels as well as a few other
things (the full list is [here](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:language.restrictions.kernels)).
No worries though, we can simply replace them with `sycl::` analogs. That's it:

```c++
sycl::event blur_image(sycl::queue queue, sycl::event waitEvent, const DeviceImage &in, DeviceImage &out) {
  sycl::range<2> rng{in.get_width(), in.get_height()};

  return queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(waitEvent);
    cgh.parallel_for(rng, [in, out](sycl::item<2> item) {
      constexpr int radius = 4;

      sycl::id<2> id = item.get_id();

      size_t i = item.get_id()[0];
      size_t j = item.get_id()[1];

      sycl::vec<unsigned int, 3> sum{0};
      for (int x = -radius; x < radius; x++) {
        for (int y = -radius; y < radius; y++) {
          constexpr size_t zero = 0; // I'm so glad C++23 will fix that.
          size_t realX = sycl::clamp(i + x, zero, in.get_width());
          size_t realY = sycl::clamp(j + y, zero, in.get_height());

          sycl::vec<unsigned int, 3> val = in(realX, realY);
          sum += val;
        }
      }

      constexpr unsigned int area = (2 * radius + 1) * (2 * radius + 1);
      sum /= area;
      out.setPixel(sum, i, j);
    });
  });
}
```

We used the simplest form of `parallel_for`. It is useful when you port your
code from a different programming model. The SYCL runtime will choose some
reasonable scheduling to achieve descent performance. But developers have much
more control over execution, and can fine-tune scheduling to yield maximum
performance.

SYCL defines its execution model as a N-dimensional index space (where N is 1,
2, or 3). Each point in this space is a kernel invokation (we refer to it as a
work-item). Work-items are grouped into work-groups. Each work-group is allowed
to make independent execution progress. Work-items inside a work-group can share
a memory region (local memory in SYCL and OpenCL, shared memory in CUDA), and
synchronize execution with barriers. Some devices support splitting work-groups
into 1-dimensional sub-groups (this can be referred to as "warp" in CUDA or
SIMD-lane on CPU programming). The following reading will help you master all
these nuances:

- [3.7.2. SYCL kernel execution model](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_kernel_execution_model)
- [4.9. Expressing parallelism through kernels](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:expr-parall-thro)

**NOTE** SYCL 1.2.1 required developers to specify kernel name as a template
parameter to `parallel_for`. You can still see code like
`cgh.parallel_for<class MyKernelName>()`. Although not required now, it is still
a valid piece code, and kernel names can be used for debugging purposes.

After the workload is completed, we need to move the result from the device
memory to our host. This can be accomplished as easy as:

```c++
int main() {
  // ...
  sycl::event cpyEvt = q.memcpy(inDev.get_pointer(), inImg.get_pointer(), 3 * inImg.get_height() * inImg.get_width());
  sycl::event blurEvt = blur_image(q, cpyEvt, inDev, outDev);

  sycl::event outEvt = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(blurEvt);
    cgh.memcpy(outImg.get_pointer(), outDev.get_pointer(), 3 * outImg.get_height() * outImg.get_width());
  });

  outEvt.wait();

  sycl::free(inDev.get_pointer(), q);
  sycl::free(outDev.get_pointer(), q);

  return 0;
}
```

Unlike our first memory copy, we don't want to use the shortcut `queue::memcpy`
here. Instead we want to submit a Command Group, that knows how to wait for our
workload to finish. Then, we wait on our data copy event. Finally, do not forget
to free your memory to avoid leaks.

Instead of using `depends_on`, we could have called `wait()` for each stage. The
disadvantage is that it is a blocking call, meaning we'd waste our CPU cycles,
waiting for the event to finish. We chose a smarter approach to let the SYCL
runtime manage our dependencies for us, and only wait when we actually need data
on host.

There's only one thing left: compile the code. Unlike your typical C++
application, SYCL application runs on multiple incompatible platforms, and must
be compiled for each device separately. Luckily, the compiler driver will handle
all this stuff for us.

The simplest compile command would look like:

```bash
clang++ -fsycl -fsycl-unnamed-lambda application.cpp -o myapp
```

This will generate a so-called fat binary, compatible with SPIR-V capable OpenCL
platforms and Level Zero backend (this covers Intel CPUs, GPUs and FPGAs).

Users of the closed-source compiler can use an even shorter form:

```bash
dpcpp application.cpp -o myapp
```

However, the open-source version (if built properly) has experimental support
for NVIDIA CUDA and AMD ROCm GPUs:

```c++
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda application.cpp -o app
clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=gfx906 -fno-sycl-libspirv -fsycl-unnamed-lambda application.cpp -o app
```

If you want your application to support multiple platforms, just specify them:
```c++
clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice,nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda application.cpp -o app
```

The full guide on compiler flags can be found [here](https://intel.github.io/llvm-docs/UsersManual.html).

There's a lot of stuff going under the hood of the compiler. It is described in
detail in our [Compiler and Runtime architecture design](https://intel.github.io/llvm-docs/CompilerAndRuntimeDesign.html)
document.

The full code is provided in [02_image_blur_usm.cpp](https://github.com/alexbatashev/get_started_with_dpcpp/blob/master/02_image_blur_usm.cpp).

## Step 3. Buffers.

In the previous example we learnt how to use USM to port our legacy code to SYCL
programming model. But SYCL runtime can also spare us the trouble of managing
device memory. In this section we will get familiar with SYCL buffers.

A buffer in SYCL is something that represents memory objects, managed by the
runtime. It is not a directly accessible piece of memory. Buffers are shared
across all devices and host. The SYCL runtime knows when to move the data
between devices and host. It can also track memory dependencies to correctly
schedule task execution.

Let's start by creating two buffers for our input and output images:

```c++
int main() {
  // ...

  sycl::range imageSize{inImg.get_width(), inImg.get_height()};
  sycl::buffer<Pixel, 2> inp{inImg.get_pointer(), imageSize};
  sycl::buffer<Pixel, 2> out{outImg.get_pointer(), imageSize};

  blur_image(q, inp, out);
}
```

In this particular case we ask SYCL runtime to use our allocated host memory for
these buffers. If SYCL runtime finds these pointers acceptable, it will use them
as-is. If not, it will allocate properly-aligned pointers and copy the data. We
could also specified a pair of iterators for our buffer, or we could skip this
parameter at all and simply create uninitialized buffer.

**IMPORTANT** It is illegal to perform any operations on memory, managed by the
buffer. Make sure the buffer is destroyed, and only then access the underlying
memory.

Also, as you can see, we created our buffers of `Pixel` type, meaning we no
longer need `DeviceImage` class.

Let's see how `blur_image` changes now. First, we no longer accept or return
events. SYCL runtime will manage dependencies for us. Next, we change the
signature to accept buffers. Our command group also changes. Instead of asking
to wait for an event, we create two accessor objects. By doing so, we ask
runtime to provide us with access to the particular buffer on a particular
device. Runtime will manage all the data movements. It will also use this
information to schedule our kernels (i.e. if there's no data dependency, two
kernels may be submitted simultaneously, leading to better device utilization).
But what if I want to access my memory on host, while keeping the buffer? Well,
there's a `host_accessor`, which will do pretty much the same for host. Note
that in this case we create one read-only and one write-only accessor, but
there're also `sycl::read_write` accessors.

We now can use these accessor objects to work with our memory. There're also
some minor changes to the kernel itself:

```c++
void blur_image(sycl::queue &q, sycl::buffer<Pixel, 2> &img, sycl::buffer<Pixel, 2> &out) {
  q.submit([&](sycl::handler &CGH) {
    sycl::accessor inp{img, CGH, sycl::read_only};
    sycl::accessor outp{out, CGH, sycl::write_only};

    sycl::range range = img.get_range();

    CGH.parallel_for(range, [=](sycl::item<2> item) {
      constexpr int radius = 4;

      sycl::id<2> id = item.get_id();
      sycl::range<2> range = item.get_range();

      sycl::vec<unsigned int, 3> sum{0};
      for (int x = -radius; x < radius; x++) {
        for (int y = -radius; y < radius; y++) {
          const auto clampedDim = [=](long val, size_t dim) {
            return static_cast<size_t>(sycl::clamp<long>(val+id[dim], 0, range[dim]));
          };
          size_t realX = clampedDim(x, 0);
          size_t realY = clampedDim(y, 1);

          sycl::vec<unsigned int, 3> val = inp[realX][realY];
          sum += val;
        }
      }

      constexpr unsigned int area = (2 * radius + 1) * (2 * radius + 1);

      outp[id] = sum / area;
    });
  });
}
```

Et voilà!

But wait, how do we get our data back? Well, `sycl::buffer` destructor is
blocking, and it is guaranteed to wait for all related tasks to complete and
copy data back to the pointer, we specified on creation. We could also use host
accessors.

By this point you may be wondering, if all this automatic memory management is
efficient at all. In many cases it is more efficient than what developers can do
on their own. This [Twitter thread](https://twitter.com/codeandrew/status/1283052563991015425)
by Codeplay's CEO Andrew Richards explains why.

The full code is provided in [03_image_blur_buffers.cpp](https://github.com/alexbatashev/get_started_with_dpcpp/blob/master/03_image_blur_buffers.cpp).

## Debugging tips

Debugging multi-threaded heterogeneous applications can be daunting. But
there're a few tips that can help you through it.

1. `sycl::stream` is analog to `std::cout`, programmers' most loved debugging
   tool. See [spec docs](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:stream).
2. Tests. SYCL is a standard C++ application. All your testing techniques still
   work. You can write Google Test or Catch unit tests to have confidence in
   your code.
3. Host device. Earlier we discussed, that SYCL can offload your code on
   accelerators. But it can also execute it on host, meaning you can use your
   favorite debugger. Just replace `default_selector` with `host_selector`.
4. oneAPI Base Toolkit comes with a oneapi-gdb tool, that allows you to debug
   DPC++ applications right on GPU. See [this nice conference talk](https://www.youtube.com/watch?v=-gQMHMdwamw)
   for overview and usage examples.

## Where to go from here?

We barely scratched the surface of accelerated computing with SYCL. There's a
lot more to learn. Below are a few sources in no particular order to pursue
this path.

1. SYCL specification: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html
2. SYCL.tech: https://sycl.tech/
3. Data Parallel C++ e-book: https://www.apress.com/gp/book/9781484255735
4. SYCLcon recordings:
   - 2020: https://www.iwocl.org/iwocl-2020/sycl-tutorials/
   - 2021: https://www.iwocl.org/iwocl-2021/conference-program/
5. List of resources, curated by James Reinders: https://jamesreinders.com/dpcpp/
6. oneAPI GPU Optimization Guide: https://software.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top/introduction.html
7. oneAPI samples: https://github.com/shuoniu-intel/oneAPI-samples/tree/master/DirectProgramming
8. oneAPI-DirectProgramming (contains benchmarks in CUDA, OpenMP, HIP and SYCL): https://github.com/zjin-lcf/oneAPI-DirectProgramming
9. Other SYCL implementations:
   - hipSYCL: https://github.com/illuhad/hipSYCL
   - ComputeCpp: https://developer.codeplay.com/products/computecpp/ce/guides/
   - triSYCL: https://github.com/triSYCL/triSYCL
10. DPC++ execution graph overview:
    - https://intel.github.io/llvm-docs/doxygen/group__sycl__graph.html
    - https://intel.github.io/llvm-docs/doxygen/classcl_1_1sycl_1_1detail_1_1Scheduler.html
