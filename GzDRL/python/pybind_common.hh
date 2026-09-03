// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#ifndef gzdrl_PYBIND_COMMON_HH_
#define gzdrl_PYBIND_COMMON_HH_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rl_server.hh"

namespace py = pybind11;

/** @brief DTypeSpec
 * structure to hold numpy dtype specifications
 */
struct DTypeSpec {
  py::dtype dtype;         // numpy dtype
  std::size_t itemsize;    // bytes per channel
  int channels;            // 1, 3, or 4
};

/** @brief Returns the numpy dtype specification for a given PixelFormatType */
inline DTypeSpec dtype_spec_for(const gz::msgs::PixelFormatType fmt) {
  switch (fmt) {
    // Luma (mono)
    case gz::msgs::PixelFormatType::L_INT8:
      return { py::dtype::of<std::uint8_t>(), 1u, 1 };
    case gz::msgs::PixelFormatType::L_INT16:
      return { py::dtype::of<std::uint16_t>(), 2u, 1 };

    // RGB family (kept in original channel order)
    case gz::msgs::PixelFormatType::RGB_INT8:
      return { py::dtype::of<std::uint8_t>(), 1u, 3 };
    case gz::msgs::PixelFormatType::RGBA_INT8:
      return { py::dtype::of<std::uint8_t>(), 1u, 4 };
    case gz::msgs::PixelFormatType::RGB_INT16:
      return { py::dtype::of<std::uint16_t>(), 2u, 3 };
    case gz::msgs::PixelFormatType::RGB_INT32:
      return { py::dtype::of<std::uint32_t>(), 4u, 3 };

    // BGR family (kept in original channel order)
    case gz::msgs::PixelFormatType::BGR_INT8:
      return { py::dtype::of<std::uint8_t>(), 1u, 3 };
    case gz::msgs::PixelFormatType::BGRA_INT8:
      return { py::dtype::of<std::uint8_t>(), 1u, 4 };
    case gz::msgs::PixelFormatType::BGR_INT16:
      return { py::dtype::of<std::uint16_t>(), 2u, 3 };
    case gz::msgs::PixelFormatType::BGR_INT32:
      return { py::dtype::of<std::uint32_t>(), 4u, 3 };

    // Float grayscale/RGB
    case gz::msgs::PixelFormatType::R_FLOAT16:
      return { py::dtype("float16"), 2u, 1 }; // why: half not a native C++ type
    case gz::msgs::PixelFormatType::R_FLOAT32:
      return { py::dtype::of<float>(), 4u, 1 };
    case gz::msgs::PixelFormatType::RGB_FLOAT16:
      return { py::dtype("float16"), 2u, 3 };
    case gz::msgs::PixelFormatType::RGB_FLOAT32:
      return { py::dtype::of<float>(), 4u, 3 };

    // Bayer 8-bit (single-channel)
    case gz::msgs::PixelFormatType::BAYER_RGGB8:
    case gz::msgs::PixelFormatType::BAYER_BGGR8:
    case gz::msgs::PixelFormatType::BAYER_GBRG8:
    case gz::msgs::PixelFormatType::BAYER_GRBG8:
      return { py::dtype::of<std::uint8_t>(), 1u, 1 };

    default:
      throw std::runtime_error("Unsupported or unknown PixelFormatType: " +
        gz::msgs::ConvertPixelFormatType(fmt));
  }
}

/** @brief Converts a gz::msgs::Image message to a contiguous NumPy array,
 * stripping row padding if necessary */
inline py::array convert_image_msg_to_numpy_copy(const gz::msgs::Image &img_msg) {

  const int width  = img_msg.width();
  const int height = img_msg.height();
  int step         = img_msg.step(); // bytes per row (may include padding)
  const auto fmt   = img_msg.pixel_format_type();
  const std::string &payload = img_msg.data();

  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid image dimensions");
  }
  py::gil_scoped_acquire gil;
  const DTypeSpec spec = dtype_spec_for(fmt);
  const std::size_t row_bytes_min = static_cast<std::size_t>(width) * spec.channels * spec.itemsize;

  // Some producers set step==0; infer when necessary.
  if (step == 0) step = static_cast<int>(row_bytes_min);

  if (static_cast<std::size_t>(step) < row_bytes_min) {
    throw std::runtime_error("img_msg.step() smaller than minimal row size");
  }

  // Bounds check on buffer length
  const std::size_t needed = static_cast<std::size_t>(height - 1) * static_cast<std::size_t>(step) + row_bytes_min;
  if (payload.size() < needed) {
    throw std::runtime_error("Image payload too small for declared size/step");
  }

  // Allocate target NumPy array (contiguous)
  const bool is_single_channel = (spec.channels == 1);
  std::vector<py::ssize_t> shape, strides; // default strides (contiguous) by not passing explicit strides
  if (is_single_channel) {
    shape = { static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width) };
  } else {
    shape = { static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(spec.channels) };
  }

  py::array out(spec.dtype, shape);
  auto *dst = static_cast<std::uint8_t*>(out.request(/*writable=*/true).ptr);

  // Fast path: already tightly packed -> single copy
  const std::size_t dst_row_bytes = row_bytes_min;
  if (static_cast<std::size_t>(step) == dst_row_bytes) {
    std::memcpy(dst, payload.data(), static_cast<std::size_t>(height) * dst_row_bytes);
    return out;
  }

  // General path: copy row-by-row, stripping padding
  const auto *src = reinterpret_cast<const std::uint8_t*>(payload.data());
  for (int r = 0; r < height; ++r) {
    std::memcpy(dst + static_cast<std::size_t>(r) * dst_row_bytes,
                src + static_cast<std::size_t>(r) * static_cast<std::size_t>(step),
                dst_row_bytes);
  }
  return out;
}

/** Keep a Python callback alive and ensure its final decref holds the GIL. */
inline std::shared_ptr<py::function> keep_python_callback(py::function callback) {
  return std::shared_ptr<py::function>(
      new py::function(std::move(callback)),
      [](py::function* value) {
        if (Py_IsInitialized()) {
          py::gil_scoped_acquire gil;
          delete value;
        } else {
          // At interpreter shutdown there is no safe runtime in which to
          // decref.  Release the handle before deleting the C++ wrapper.
          value->release();
          delete value;
        }
      });
}

/** @brief Checks if a vector is broadcastable to a given size n */
// why: consistent broadcasting semantics for all vectorized args.
template <typename T>
inline void check_broadcastable(std::size_t n, const std::vector<T>& v, const char* name) {
  if (!(v.size() == 1 || v.size() == n)) {
    throw py::value_error(std::string("Argument '") + name + "' must have size 1 or match env_ids size (" +
                          std::to_string(n) + "), got " + std::to_string(v.size()));
  }
}

/** @brief Picks an element from a vector, supporting broadcasting semantics */
template <typename T>
inline const T& pick(const std::vector<T>& v, std::size_t i) {
  return v.size() == 1 ? v[0] : v[i];
}

#endif  // gzdrl_PYBIND_COMMON_HH_
