#include <CL/sycl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <filesystem>
#include <stdexcept>
#include <string>

using byte = unsigned char;

class Pixel {
public:
  Pixel() = default;

  Pixel(byte r, byte g, byte b) : r(r), g(g), b(b) {}

  Pixel(const sycl::vec<unsigned int, 3> &inp) { fromArray(inp); }

  Pixel &operator=(sycl::vec<unsigned int, 3> inp) {
    fromArray(inp);
    return *this;
  }

  operator sycl::vec<unsigned int, 3>() const noexcept { return {r, g, b}; }

  byte getR() const noexcept { return r; }
  byte getG() const noexcept { return g; }
  byte getB() const noexcept { return b; }

private:
  void fromArray(const sycl::vec<unsigned int, 3> &inp) {
    sycl::vec<unsigned int, 3> minVal{0, 0, 0};
    sycl::vec<unsigned int, 3> maxVal{255u, 255u, 255u};

    sycl::vec<byte, 3> data = sycl::clamp(inp, minVal, maxVal).convert<byte>();

    r = data.s0();
    g = data.s1();
    b = data.s2();
  }

  byte r, g, b;
};

class Image {
public:
  enum class mode { read_only, create };

  Image(std::filesystem::path path, mode mode = mode::read_only) : mMode(mode) {
    assert(mMode != mode::create);
    int inputChannels;

    constexpr int requiredChannels = 3;

    mData = stbi_load(path.c_str(), &mWidth, &mHeight, &inputChannels,
                      requiredChannels);

    if (!mData)
      throw std::runtime_error{"Failed to load image from " + path.string()};
  }

  Image(std::filesystem::path path, size_t width, size_t height,
        mode mode = mode::create)
      : mMode(mode), mWidth(static_cast<int>(width)),
        mHeight(static_cast<int>(height)), mPath(path) {
    assert(mMode == mode::create);

    constexpr int channels = 3;

    mData = new byte[width * height * channels];
    if (!mData)
      throw std::runtime_error{"Failed to create image"};
  }
  ~Image() {
    if (mMode == mode::create) {
      sycl::vec<unsigned int, 3> v = *static_cast<Pixel *>(mData);
      stbi_write_png(mPath.c_str(), mWidth, mHeight, 3, mData, 0);
      delete[] static_cast<byte *>(mData);
    } else {
      stbi_image_free(mData);
    }
  }

  size_t get_width() const noexcept { return mWidth; }
  size_t get_height() const noexcept { return mHeight; }

  const Pixel *get_pointer() const noexcept {
    return static_cast<Pixel *>(mData);
  }
  Pixel *get_pointer() noexcept { return static_cast<Pixel *>(mData); }

private:
  mode mMode;
  int mWidth;
  int mHeight;
  std::filesystem::path mPath;
  void *mData = nullptr;
};

void blur_image(sycl::queue &q, sycl::buffer<Pixel, 2> &img,
                sycl::buffer<Pixel, 2> &out) {
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
            return static_cast<size_t>(
                sycl::clamp<long>(val + id[dim], 0, range[dim]));
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

int main(int argc, char *argv[]) {
  const Image inImg(std::filesystem::path{argv[1]});
  Image outImg("blurred_buf.png", inImg.get_width(), inImg.get_height());

  sycl::queue q{sycl::default_selector{}};

  sycl::range imageSize{inImg.get_width(), inImg.get_height()};
  sycl::buffer<Pixel, 2> inp{inImg.get_pointer(), imageSize};
  sycl::buffer<Pixel, 2> out{outImg.get_pointer(), imageSize};

  blur_image(q, inp, out);

  return 0;
}
