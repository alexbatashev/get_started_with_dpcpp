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

  Pixel(const std::array<unsigned int, 3> &inp) { fromArray(inp); }

  Pixel &operator=(std::array<unsigned int, 3> inp) {
    fromArray(inp);
    return *this;
  }

  operator std::array<unsigned int, 3>() const noexcept { return {r, g, b}; }

  byte getR() const noexcept { return r; }
  byte getG() const noexcept { return g; }
  byte getB() const noexcept { return b; }

private:
  void fromArray(const std::array<unsigned int, 3> &inp) {
    constexpr unsigned int minVal = 0;
    constexpr unsigned int maxVal = 255u;

    r = static_cast<byte>(std::clamp(inp[0], minVal, maxVal));
    g = static_cast<byte>(std::clamp(inp[1], minVal, maxVal));
    b = static_cast<byte>(std::clamp(inp[2], minVal, maxVal));
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

void blur_image(const Image &in, Image &out) {
#pragma parallel for collapse(2)
  for (size_t i = 0; i < in.get_width(); i++) {
    for (size_t j = 0; j < in.get_height(); j++) {
      constexpr int radius = 4;

      std::array<unsigned int, 3> sum{0};
      for (int x = -radius; x < radius; x++) {
        for (int y = -radius; y < radius; y++) {
          constexpr size_t zero = 0; // I'm so glad C++23 will fix that.
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

int main(int argc, char *argv[]) {
  const Image inImg(std::filesystem::path{argv[1]});
  Image outImg("blurred_omp.png", inImg.get_width(), inImg.get_height());

  blur_image(inImg, outImg);

  return 0;
}
