#include <CL/sycl.hpp>

int main() {
  // static auto exception_handler = [](sycl::exception_list e_list) {
  //   for (std::exception_ptr const &e : e_list) {
  //     try {
  //       std::rethrow_exception(e);
  //     } catch (std::exception const &e) {
  //       std::terminate();
  //     }
  //   }
  // };

  sycl::queue q;
  // sycl::queue q{sycl::default_selector{}, exception_handler};

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int size = 10000;

  int *a = sycl::malloc_shared<int>(size, q);
  int *b = sycl::malloc_shared<int>(size, q);
  int *result = sycl::malloc_shared<int>(size, q);

  auto initialize_array = [&size](int *array) {
    for (int i = 0; i < size; i++) {
      array[i] = i;
    }
  };

  initialize_array(a);
  initialize_array(b);

  sycl::range<1> num_items{size};

  auto e = q.parallel_for(num_items, [=](auto i) { result[i] = a[i] + b[i]; });

  auto e = q.submit([&](sycl::handler &cgh) {
    sycl::stream Out(1024, 80, cgh);
    cgh.parallel_for<class MyVectorAddKernel>(num_items, [=](sycl::id<1> i) {
      result[i] = a[i] + b[i] + foo(i.get(0));
      if (i < 5) {
        Out << "result[" << i.get(0) << "] == " << result[i] << "\n";
      }
    });
  });

  e.wait();

  for (int i = 0; i < size; i++) {
    if (result[i] != a[i] + b[i]) {
      std::cerr << "ERROR: result[" << i << "] == " << result[i] << "\n";
      return -1;
    }
  }

  std::cout << "Results correct!\n";

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(result, q);

  return EXIT_SUCCESS;
}
