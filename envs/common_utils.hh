// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#pragma once
#ifndef COMMON_UTILS_HH_
#define COMMON_UTILS_HH_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <Eigen/Dense>
#include <gz/msgs/image.pb.h>

/// Serialize Gazebo server construction across every environment type.
inline std::mutex gzdrl_gazebo_construction_mutex;


/**
 * @brief Thread-safe random number generator using uniform distribution.
 *
 * This class provides a simple interface for generating random numbers within
 * a specified range. It uses the Mersenne Twister engine (mt19937) with a
 * uniform real distribution.
 *
 * @tparam T Numeric type for the random numbers (e.g., float, double, int)
 */
template <typename T>
class RNG
{
public:
  /**
   * @brief Constructs an RNG with specified bounds and seed.
   *
   * @param lower_lim Lower bound of the distribution (inclusive)
   * @param upper_lim Upper bound of the distribution (inclusive)
   * @param seed Seed value for deterministic random number generation
   */
  RNG(T lower_lim, T upper_lim, float seed)
      : lower_lim_(lower_lim),
        upper_lim_(upper_lim),
        dev_(std::make_unique<std::random_device>()),
        rng_(std::make_unique<std::mt19937>((*dev_)())),
        dist_(std::make_unique<std::uniform_real_distribution<>>(
            static_cast<double>(lower_lim),
            static_cast<double>(upper_lim)))
  {
    // Use provided seed for deterministic behavior
    rng_->seed(static_cast<unsigned int>(seed));
  }

  /**
   * @brief Generates N random samples.
   *
   * This method is thread-safe due to the internal mutex lock.
   *
   * @param N Number of samples to generate
   * @return std::vector<T> Vector containing N random samples
   */
  std::vector<T> sample(int N)
  {
    std::lock_guard<std::mutex> guard(lock_);
    std::vector<T> res;
    res.reserve(N); // Pre-allocate for efficiency

    for (int i = 0; i < N; ++i)
    {
      res.push_back(static_cast<T>((*dist_)(*rng_)));
    }

    return res;
  }
  /** @brief update the limits */
  void update_limits(T lower_lim, T upper_lim)
  {
    std::lock_guard<std::mutex> guard(lock_);
    lower_lim_ = lower_lim;
    upper_lim_ = upper_lim;
    dist_ = std::make_unique<std::uniform_real_distribution<>>(
        static_cast<double>(lower_lim_),
        static_cast<double>(upper_lim_));
  }

private:
  std::mutex lock_;                                        ///< Mutex for thread safety
  T lower_lim_;                                            ///< Lower bound of distribution
  T upper_lim_;                                            ///< Upper bound of distribution
  std::unique_ptr<std::random_device> dev_;                ///< Random device for seeding
  std::unique_ptr<std::mt19937> rng_;                      ///< Mersenne Twister RNG engine
  std::unique_ptr<std::uniform_real_distribution<>> dist_; ///< Uniform distribution
};

/**
 * @brief A fixed-capacity circular buffer (ring buffer) implementation.
 *
 * This ring buffer provides two modes of insertion:
 * 1. push() - Fails when buffer is full
 * 2. push_overwrite() - Overwrites oldest element when buffer is full
 *
 * The buffer maintains insertion order and provides FIFO semantics.
 * Elements can be peeked from the latest end without removal.
 *
 * @tparam T Type of elements stored in the buffer
 */
template <class T>
class RingBuffer
{
public:
  /**
   * @brief Constructs a ring buffer with specified capacity.
   *
   * @param capacity Maximum number of elements the buffer can hold
   * @throws std::invalid_argument if capacity is zero
   */
  explicit RingBuffer(size_t capacity) : capacity_(capacity), write_idx_(0), size_(0)
  {
    if (capacity == 0)
    {
      throw std::invalid_argument("RingBuffer capacity must be greater than 0");
    }
    buffer_.resize(capacity);
  }

  // Prevent copying to avoid issues with shared state
  RingBuffer(const RingBuffer &) = delete;
  RingBuffer &operator=(const RingBuffer &) = delete;

  /**
   * @brief Pushes an element, overwriting the oldest if buffer is full.
   *
   * This method never fails. If the buffer is at capacity, it removes
   * the oldest element before inserting the new one.
   *
   * @tparam U Type of value being pushed (forwarding reference)
   * @param value Value to insert into the buffer
   */
  template <class U>
  void push_overwrite(U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>)
  {
    buffer_[write_idx_] = std::forward<U>(value);
    write_idx_ = (write_idx_ + 1) % capacity_;

    // Increase size only until we reach capacity
    if (size_ < capacity_)
    {
      ++size_;
    }
  }

  /**
   * @brief Attempts to push an element without overwriting.
   *
   * If the buffer is full, this method returns false and does not
   * insert the element.
   *
   * @tparam U Type of value being pushed (forwarding reference)
   * @param value Value to insert into the buffer
   * @return true if insertion succeeded, false if buffer was full
   */
  template <class U>
  bool push(U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>)
  {
    if (size_ >= capacity_)
    {
      return false; // Buffer is full
    }

    buffer_[write_idx_] = std::forward<U>(value);
    write_idx_ = (write_idx_ + 1) % capacity_;
    ++size_;

    return true;
  }

  /**
   * @brief Removes and returns the oldest element from the buffer.
   *
   * @return std::optional<T> The oldest element, or std::nullopt if buffer is empty
   */
  std::optional<T> pop() noexcept(std::is_nothrow_move_constructible_v<T>)
  {
    if (size_ == 0)
    {
      return std::nullopt;
    }

    // Calculate read index: the oldest element is at (write_idx - size)
    size_t read_idx = (write_idx_ + capacity_ - size_) % capacity_;
    T value = std::move(buffer_[read_idx]);
    --size_;

    return value;
  }

  /**
   * @brief Checks if the buffer is empty.
   * @return true if buffer contains no elements
   */
  bool empty() const noexcept
  {
    return size_ == 0;
  }

  /**
   * @brief Checks if the buffer is at full capacity.
   * @return true if buffer cannot accept more elements without overwriting
   */
  bool full() const noexcept
  {
    return size_ == capacity_;
  }

  /**
   * @brief Returns the current number of elements in the buffer.
   * @return Current size
   */
  size_t size() const noexcept
  {
    return size_;
  }

  /**
   * @brief Returns the maximum capacity of the buffer.
   * @return Maximum capacity
   */
  size_t capacity() const noexcept
  {
    return capacity_;
  }

  /**
   * @brief Copies up to N of the most recent elements to an output iterator.
   *
   * Elements are copied in chronological order (oldest to newest).
   * Does not remove elements from the buffer.
   *
   * @tparam OutputIt Output iterator type
   * @param N Maximum number of elements to copy
   * @param out Output iterator where elements will be written
   * @return Number of elements actually copied
   */
  template <class OutputIt>
  size_t peek_latest(size_t N, OutputIt out) const
  {
    if (size_ == 0 || N == 0)
    {
      return 0;
    }

    // Copy at most min(N, size_) elements
    const size_t count = std::min(N, size_);

    // Start from the (size_ - count)th oldest element
    const size_t start_offset = size_ - count;
    const size_t start_idx = (write_idx_ + capacity_ - size_ + start_offset) % capacity_;

    // Copy elements in order
    for (size_t i = 0; i < count; ++i)
    {
      const size_t idx = (start_idx + i) % capacity_;
      *out++ = buffer_[idx];
    }

    return count;
  }

  /**
   * @brief Copies up to N of the most recent elements in reverse order.
   *
   * Elements are copied from newest to oldest.
   * Does not remove elements from the buffer.
   *
   * @tparam OutputIt Output iterator type
   * @param N Maximum number of elements to copy
   * @param out Output iterator where elements will be written
   * @return Number of elements actually copied
   */
  template <class OutputIt>
  size_t peek_latest_reverse(size_t N, OutputIt out) const
  {
    if (size_ == 0 || N == 0)
    {
      return 0;
    }

    // Copy at most min(N, size_) elements
    const size_t count = std::min(N, size_);

    // Start from the most recent element (just before write_idx_)
    for (size_t i = 0; i < count; ++i)
    {
      const size_t idx = (write_idx_ + capacity_ - 1 - i) % capacity_;
      *out++ = buffer_[idx];
    }

    return count;
  }

private:
  const size_t capacity_; ///< Maximum number of elements
  std::vector<T> buffer_; ///< Underlying storage
  size_t write_idx_;      ///< Index where next element will be written
  size_t size_;           ///< Current number of elements in buffer
};


/**
 * @brief Converts a Gazebo image message to RGB888 format (HWC layout).
 *
 * This function handles various pixel formats from Gazebo and converts them
 * to a standard RGB888 format with HWC (Height-Width-Channel) layout.
 * Properly handles row stride (step) to account for padding.
 *
 * Supported formats:
 * - RGB_INT8: 8-bit RGB
 * - BGR_INT8: 8-bit BGR (swapped to RGB)
 * - RGBA_INT8: 8-bit RGBA (alpha dropped)
 * - BGRA_INT8: 8-bit BGRA (swapped to RGB, alpha dropped)
 * - L_INT8: 8-bit grayscale (replicated to RGB)
 *
 * @param img Gazebo image message
 * @return std::vector<uint8_t> RGB888 buffer (H * W * 3 bytes)
 */
inline std::vector<uint8_t> ToRGB8_HWC(const gz::msgs::Image &img)
{
  const uint32_t w = img.width();
  const uint32_t h = img.height();
  const uint32_t step = img.step(); // Bytes per row (may include padding)
  const auto fmt = img.pixel_format_type();
  const std::string &blob = img.data();
  const auto *src = reinterpret_cast<const uint8_t *>(blob.data());

  // Validate input dimensions and data size
  if (blob.size() < static_cast<size_t>(h) * step || w == 0 || h == 0)
  {
    // Return solid black image if input is invalid
    return std::vector<uint8_t>(static_cast<size_t>(h) * w * 3, 0);
  }

  // Allocate output buffer (contiguous RGB888, no padding)
  std::vector<uint8_t> out(static_cast<size_t>(h) * w * 3);

  // Lambda to get destination row pointer
  auto line_dst = [&](uint32_t y) -> uint8_t *
  {
    return &out[static_cast<size_t>(y) * w * 3];
  };

  // Lambda to get source row pointer (accounting for stride)
  auto line_src = [&](uint32_t y) -> const uint8_t *
  {
    return src + static_cast<size_t>(y) * step;
  };

  switch (fmt)
  {
  case gz::msgs::PixelFormatType::RGB_INT8:
  {
    // Native RGB format - fast path with memcpy if no padding
    if (step == 3 * w)
    {
      // No padding: single memcpy for entire image
      std::memcpy(out.data(), src, static_cast<size_t>(h) * w * 3);
    }
    else
    {
      // Padding present: copy row by row
      for (uint32_t y = 0; y < h; ++y)
      {
        std::memcpy(line_dst(y), line_src(y), static_cast<size_t>(w) * 3);
      }
    }
    break;
  }

  case gz::msgs::PixelFormatType::BGR_INT8:
  {
    // BGR format: swap R and B channels
    for (uint32_t y = 0; y < h; ++y)
    {
      const uint8_t *s = line_src(y);
      uint8_t *d = line_dst(y);
      for (uint32_t x = 0; x < w; ++x)
      {
        const uint8_t *p = s + x * 3;
        d[x * 3 + 0] = p[2]; // R = B
        d[x * 3 + 1] = p[1]; // G = G
        d[x * 3 + 2] = p[0]; // B = R
      }
    }
    break;
  }

  case gz::msgs::PixelFormatType::RGBA_INT8:
  {
    // RGBA format: drop alpha channel
    for (uint32_t y = 0; y < h; ++y)
    {
      const uint8_t *s = line_src(y);
      uint8_t *d = line_dst(y);
      for (uint32_t x = 0; x < w; ++x)
      {
        const uint8_t *p = s + x * 4;
        d[x * 3 + 0] = p[0]; // R
        d[x * 3 + 1] = p[1]; // G
        d[x * 3 + 2] = p[2]; // B (alpha at p[3] is dropped)
      }
    }
    break;
  }

  case gz::msgs::PixelFormatType::BGRA_INT8:
  {
    // BGRA format: swap R and B, drop alpha
    for (uint32_t y = 0; y < h; ++y)
    {
      const uint8_t *s = line_src(y);
      uint8_t *d = line_dst(y);
      for (uint32_t x = 0; x < w; ++x)
      {
        const uint8_t *p = s + x * 4;
        d[x * 3 + 0] = p[2]; // R = B
        d[x * 3 + 1] = p[1]; // G = G
        d[x * 3 + 2] = p[0]; // B = R (alpha at p[3] is dropped)
      }
    }
    break;
  }

  case gz::msgs::PixelFormatType::L_INT8:
  {
    // Grayscale: replicate luminance to all three RGB channels
    for (uint32_t y = 0; y < h; ++y)
    {
      const uint8_t *s = line_src(y);
      uint8_t *d = line_dst(y);
      for (uint32_t x = 0; x < w; ++x)
      {
        const uint8_t v = s[x];
        d[x * 3 + 0] = v; // R = L
        d[x * 3 + 1] = v; // G = L
        d[x * 3 + 2] = v; // B = L
      }
    }
    break;
  }

  default:
  {
    // Unsupported format: fill with black
    std::fill(out.begin(), out.end(), 0);
    break;
  }
  }

  return out;
}

/**
 * @brief Resizes an RGB888 image using nearest-neighbor interpolation.
 *
 * This is a fast, dependency-free resizing method suitable for real-time
 * applications. Nearest-neighbor interpolation may produce blocky results
 * but is computationally efficient.
 *
 * @param src Source image data (RGB888, HWC layout)
 * @param sw Source width
 * @param sh Source height
 * @param dst Destination buffer (must be pre-allocated to dw * dh * 3 bytes)
 * @param dw Destination width
 * @param dh Destination height
 */
inline void ResizeNearestRGB888(const uint8_t *src, int sw, int sh,
                                uint8_t *dst, int dw, int dh)
{
  // Calculate scaling factors
  const float sx = static_cast<float>(sw) / static_cast<float>(dw);
  const float sy = static_cast<float>(sh) / static_cast<float>(dh);

  // Process each destination pixel
  for (int y = 0; y < dh; ++y)
  {
    // Find corresponding source row (clamped to valid range)
    const int yy = std::min(static_cast<int>(y * sy), sh - 1);
    const uint8_t *srow = src + static_cast<size_t>(yy) * sw * 3;
    uint8_t *drow = dst + static_cast<size_t>(y) * dw * 3;

    for (int x = 0; x < dw; ++x)
    {
      // Find corresponding source column (clamped to valid range)
      const int xx = std::min(static_cast<int>(x * sx), sw - 1);
      const size_t si = static_cast<size_t>(xx) * 3;
      const size_t di = static_cast<size_t>(x) * 3;

      // Copy RGB triplet
      drow[di + 0] = srow[si + 0]; // R
      drow[di + 1] = srow[si + 1]; // G
      drow[di + 2] = srow[si + 2]; // B
    }
  }
}

#endif // COMMON_UTILS_HH_
