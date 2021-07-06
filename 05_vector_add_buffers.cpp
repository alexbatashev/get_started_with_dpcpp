#include <CL/sycl.hpp>

#include <numeric>

int main() {
  constexpr int size = 10000;

  std::vector<int> a_vec(size);
  std::vector<int> b_vec(size);
  std::vector<int> result_vec(size, 0);

  std::iota(std::begin(a_vec), std::end(a_vec), 1);
  b_vec = a_vec;

  static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const &e : e_list) {
      try {
        std::rethrow_exception(e);
      } catch (std::exception const &e) {
        std::terminate();
      }
    }
  };

  sycl::queue q{exception_handler};

  {
    sycl::buffer a_buf(a_vec);
    sycl::buffer b_buf(b_vec);
    sycl::buffer result_buf(result_vec);

    sycl::range<1> num_items{size};

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor a_acc(a_buf, cgh, sycl::read_only);
      sycl::accessor b_acc(b_buf, cgh, sycl::read_only);
      sycl::accessor result_acc(result_buf, cgh, sycl::write_only);

      cgh.parallel_for(num_items,
                       [=](auto i) { result_acc[i] = a_acc[i] + b_acc[i]; });
    });

    // example of ND-range - matrix mult
    // q.submit([&](sycl::handler &cgh) {
    //   sycl::accessor a_acc(a_buf, cgh, sycl::read_only);
    //   sycl::accessor b_acc(b_buf, cgh, sycl::read_only);
    //   sycl::accessor result_acc(result_buf, cgh, sycl::write_only);

    //   sycl::range global{100, 100};
    //   sycl::range local{10, 10};
    //   cgh.parallel_for(sycl::nd_range{global, local}, [=](nd_item<2> i) {
    //     int jj = i.get_global_id(0);
    //     int ii = i.get_global_id(1);

    //     for (int k = 0; k < N; k++) {
    //       result_acc[j][i] += a_acc[j][k] * b_acc[k][i];
    //     }
    //   });
    // });
  }

  for (int i = 0; i < size; i++) {
    if (result_vec[i] != a_vec[i] + b_vec[i]) {
      std::cerr << "ERROR: result[" << i << "] == " << result_vec[i] << "\n";
      return -1;
    }
  }

  std::cout << "Results correct!\n";

  return EXIT_SUCCESS;
}