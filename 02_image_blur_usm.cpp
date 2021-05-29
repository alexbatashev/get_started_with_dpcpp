#include <CL/sycl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <algorithm>
#include <array>
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

  Pixel operator()(size_t x, size_t y) const noexcept {
    byte *bytes = static_cast<byte *>(mData) + mWidth * x * 3 + y * 3;
    return Pixel(bytes[0], bytes[1], bytes[2]);
  }

  void setPixel(Pixel p, size_t x, size_t y) {
    byte *bytes = static_cast<byte *>(mData) + mWidth * x * 3 + y * 3;
    bytes[0] = p.getR();
    bytes[1] = p.getG();
    bytes[2] = p.getB();
  }

private:
  mode mMode;
  int mWidth;
  int mHeight;
  std::filesystem::path mPath;
  void *mData = nullptr;
};

class DeviceImage {
public:
  DeviceImage(sycl::queue q, size_t width, size_t height)
      : mWidth(width), mHeight(height) {
    mData = sycl::malloc_device<byte>(width * height * 3, q);
  }

  byte *get_pointer() noexcept { return mData; }

  const byte *get_pointer() const noexcept { return mData; }

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

sycl::event blur_image(sycl::queue queue, sycl::event waitEvent,
                       const DeviceImage &in, DeviceImage &out) {
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

int main(int argc, char *argv[]) {
  const Image inImg(std::filesystem::path{argv[1]});
  Image outImg("blurred_usm.png", inImg.get_width(), inImg.get_height());

  sycl::queue q{sycl::default_selector{}};

  DeviceImage inDev{q, inImg.get_width(), inImg.get_height()};
  DeviceImage outDev{q, inImg.get_width(), inImg.get_height()};

  sycl::event cpyEvt = q.memcpy(inDev.get_pointer(), inImg.get_pointer(),
                                3 * inImg.get_height() * inImg.get_width());
  sycl::event blurEvt = blur_image(q, cpyEvt, inDev, outDev);

  sycl::event outEvt = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(blurEvt);
    cgh.memcpy(outImg.get_pointer(), outDev.get_pointer(),
               3 * outImg.get_height() * outImg.get_width());
  });

  outEvt.wait();

  sycl::free(inDev.get_pointer(), q);
  sycl::free(outDev.get_pointer(), q);

  return 0;
}
