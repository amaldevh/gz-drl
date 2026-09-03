// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Amal Dev Haridevan

#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <gz/msgs/image.pb.h>
#if ROS_VER == 1
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
using PointFieldMsg = sensor_msgs::PointField;
#elif ROS_VER == 2
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
using PointFieldMsg = sensor_msgs::msg::PointField;
#endif

namespace sensor_converters
{
  /**
   * @brief Struct that holds the specs for a Pixeltype
   *
   */
  struct PixelSpec
  {
    std::string encoding;       ///< encoding ofd the data
    uint32_t channels;          ///< channels in the Image
    uint32_t bytes_per_channel; ///< Number of bytes per channel
  };

  /**
   * @brief Get the PixelSpec struct from the gz::msgs::PixelFormatType
   *
   * @param fmt pixel format type from gz::msgs::Image
   * @return PixelSpec the pixel spec
   */
  inline PixelSpec pixel_spec_from_gz(const gz::msgs::PixelFormatType fmt)
  {
    using PF = gz::msgs::PixelFormatType;
    switch (fmt)
    {
    case PF::L_INT8:
      return {"mono8", 1, 1};
    case PF::L_INT16:
      return {"mono16", 1, 2};
    case PF::RGB_INT8:
      return {"rgb8", 3, 1};
    case PF::RGBA_INT8:
      return {"rgba8", 4, 1};
    case PF::BGRA_INT8:
      return {"bgra8", 4, 1};
    case PF::BGR_INT8:
      return {"bgr8", 3, 1};
    case PF::RGB_INT16:
      return {"", 3, 2}; // No standard ROS encoding.
    case PF::BGR_INT16:
      return {"", 3, 2};
    case PF::RGB_INT32:
      return {"", 3, 4};
    case PF::BGR_INT32:
      return {"", 3, 4};
    case PF::R_FLOAT16:
      return {"16FC1", 1, 2};
    case PF::RGB_FLOAT16:
      return {"16FC3", 3, 2};
    case PF::R_FLOAT32:
      return {"32FC1", 1, 4};
    case PF::RGB_FLOAT32:
      return {"32FC3", 3, 4};
    case PF::BAYER_RGGB8:
      return {"bayer_rggb8", 1, 1};
    case PF::BAYER_BGGR8:
      return {"bayer_bggr8", 1, 1};
    case PF::BAYER_GBRG8:
      return {"bayer_gbrg8", 1, 1};
    case PF::BAYER_GRBG8:
      return {"bayer_grbg8", 1, 1};
    case PF::UNKNOWN_PIXEL_FORMAT:
    default:
      return {"", 3, 4};
      //   throw std::invalid_argument("Unsupported/unknown gz::msgs::PixelFormatType");
    }
  }

  /**
   * @brief Converts gz::msgs::Image to sensor_msgs::Image
   *
   * @param gz_img
   * @param ros
   */
  inline void convert_img(
      const gz::msgs::Image &gz_img,
      ImageMsg &ros)
  {

    const auto spec = pixel_spec_from_gz(gz_img.pixel_format_type());
    const uint32_t width = gz_img.width();
    const uint32_t height = gz_img.height();
    if (width == 0 || height == 0)
    {
      // sensor_msgs::msg::Image ros;
      // ros.header       = header;
      return;
      // return ros;
    }

    const auto &src = gz_img.data();
    const size_t expected_bytes =
        static_cast<size_t>(width) * height * spec.channels * spec.bytes_per_channel;
    if (src.size() != expected_bytes)
    {
      std::cout << "gz image data size mismatch vs (w*h*channels*bpc)\n";
      return;
    }

    //   sensor_msgs::msg::Image ros;
    //   ros.header       = header;
    ros.height = height;
    ros.width = width;
    ros.encoding = spec.encoding.empty() ? "unknown" : spec.encoding;
    ros.is_bigendian = 0;
    ros.step = width * spec.channels * spec.bytes_per_channel;
    ros.data.resize(src.size());
    std::memcpy(ros.data.data(), src.data(), src.size());
    //   return ros;
  }

  /**
   * @brief Create a float32 field msg
   *
   * @param name name of the field
   * @param offset offset
   * @return PointFieldMsg
   */
  inline PointFieldMsg make_float32_field(
      const std::string &name, uint32_t offset)
  {
    PointFieldMsg f;
    f.name = name;
    f.offset = offset;
    f.datatype = PointFieldMsg::FLOAT32;
    f.count = 1;
    return f;
  }

  /**
   * @brief Converts a gz GpuLidar frame (polar range image) to a PointCloud2 msg
   *
   * @details The gz GpuLidar callback delivers a polar range image, not Cartesian
   * points: each reading is [range, retro/intensity, unused] and the beam direction
   * is implied by the row/column position. This projects every beam into Cartesian
   * x,y,z (sensor frame, x forward) using the scan's azimuth/elevation bounds, and
   * carries the retro value through as an `intensity` field.
   *
   * @param frame LidarFrameView (range image + angle bounds)
   * @param cloud Pointcloud2 msg
   * @param organized organize as h, w instead of a single array
   */
  inline void convert_lidar(
      const LidarFrameView &frame,
      PointCloudMsg &cloud,
      bool organized = true)
  {
    if (!frame.data)
      return;
    if (frame.w == 0 || frame.h == 0 || frame.c == 0)
      return;
    if (frame.c < 3)
      return;

    const uint32_t w = frame.w;
    const uint32_t h = frame.h;
    const size_t npts = static_cast<size_t>(w) * h;

    // Per-beam angle steps. Bounds are inclusive, so divide by (count - 1).
    const float h_inc = (w > 1) ? (frame.angle_max - frame.angle_min) / static_cast<float>(w - 1) : 0.f;
    const float v_inc = (h > 1) ? (frame.vert_angle_max - frame.vert_angle_min) / static_cast<float>(h - 1) : 0.f;

    if (organized)
    {
      cloud.height = h;
      cloud.width = w;
    }
    else
    {
      cloud.height = 1;
      cloud.width = static_cast<uint32_t>(npts);
    }
    cloud.is_bigendian = false;

    // Fields: x,y,z,intensity (all float32).
    cloud.fields.clear();
    uint32_t offset = 0;
    cloud.fields.emplace_back(make_float32_field("x", offset));
    offset += 4;
    cloud.fields.emplace_back(make_float32_field("y", offset));
    offset += 4;
    cloud.fields.emplace_back(make_float32_field("z", offset));
    offset += 4;
    cloud.fields.emplace_back(make_float32_field("intensity", offset));
    offset += 4;

    cloud.point_step = offset;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(static_cast<size_t>(cloud.point_step) * cloud.width * cloud.height);

    const size_t src_stride = frame.c; // floats per reading
    bool all_finite = true;

    // row -> elevation (vertical), col -> azimuth (horizontal)
    auto write_point = [&](size_t dst_index, uint32_t row, uint32_t col, const float *src_base)
    {
      const float range = src_base[0];
      const float intensity = src_base[1];

      float x, y, z;
      if (std::isfinite(range))
      {
        const float azimuth = frame.angle_min + static_cast<float>(col) * h_inc;
        const float elevation = frame.vert_angle_min + static_cast<float>(row) * v_inc;
        const float cos_e = std::cos(elevation);
        x = range * cos_e * std::cos(azimuth);
        y = range * cos_e * std::sin(azimuth);
        z = range * std::sin(elevation);
      }
      else
      {
        // Out-of-range/no return: emit NaN so the cloud is non-dense.
        x = y = z = std::numeric_limits<float>::quiet_NaN();
        all_finite = false;
      }

      uint8_t *dst = cloud.data.data() + dst_index * cloud.point_step;
      std::memcpy(dst + cloud.fields[0].offset, &x, 4);
      std::memcpy(dst + cloud.fields[1].offset, &y, 4);
      std::memcpy(dst + cloud.fields[2].offset, &z, 4);
      std::memcpy(dst + cloud.fields[3].offset, &intensity, 4);
    };

    for (uint32_t row = 0; row < h; ++row)
    {
      for (uint32_t col = 0; col < w; ++col)
      {
        const size_t src_idx = static_cast<size_t>(row) * w + col;
        const size_t dst_idx = organized ? (static_cast<size_t>(row) * cloud.width + col) : src_idx;
        write_point(dst_idx, row, col, frame.data + src_idx * src_stride);
      }
    }

    cloud.is_dense = all_finite;
  }

  /**
   * @brief Get the camera instrinsics from gz::msgs::CameraInfo msg
   *
   * @param info camera info msg
   * @param fx focal length in x(mutable reference)
   * @param fy focal length in y (mutable reference)
   * @param cx center of camera offset in x (mutable reference)
   * @param cy center of camera offset in y (mutable reference)
   */
  inline void intrinsics_from_camera_info(const gz::msgs::CameraInfo &info,
                                          double &fx, double &fy, double &cx, double &cy)
  {
    // Row-major 3x3 K: [ fx 0 cx ; 0 fy cy ; 0 0 1 ]
    const auto &k = info.intrinsics().k();
    if (k.size() < 9)
      throw std::invalid_argument(
          "convert_depth_to_pointcloud2: CameraInfo.intrinsics.k has " +
          std::to_string(k.size()) + " entries, expected 9");
    fx = k.Get(0);
    fy = k.Get(4);
    cx = k.Get(2);
    cy = k.Get(5);
  }

  /**
   * @brief Converts depth Image to pointcloud2 msg
   *
   * @param gz_img gz::msgs::Image img
   * @param info camera info, gz::msgs::CameraInfo
   * @param cloud PointCloud2 msg
   * @param organized pack into oraganized h and w
   * @param invalid_value invalid value to use
   */
  inline void convert_depth_to_pointcloud2(
      const gz::msgs::Image &gz_img,
      const gz::msgs::CameraInfo &info,
      PointCloudMsg &cloud,
      bool organized = true,
      float invalid_value = std::numeric_limits<float>::quiet_NaN())
  {
    using PF = gz::msgs::PixelFormatType;

    // --- Enforce 32FC1 (R_FLOAT32): single-channel float32 depth in meters. ---
    if (gz_img.pixel_format_type() != PF::R_FLOAT32)
    {
      const auto spec = pixel_spec_from_gz(gz_img.pixel_format_type());
      throw std::invalid_argument(
          "convert_depth_to_pointcloud2: expected 32FC1 (R_FLOAT32) depth image, got '" +
          (spec.encoding.empty() ? std::string("unknown") : spec.encoding) + "'");
    }

    const uint32_t width = gz_img.width();
    const uint32_t height = gz_img.height();
    if (width == 0 || height == 0)
      return;

    const auto &src = gz_img.data();
    const size_t expected_bytes = static_cast<size_t>(width) * height * 4; // 1ch * 4B
    if (src.size() != expected_bytes)
    {
      std::cout << "gz depth image data size mismatch vs (w*h*4)\n";
      return;
    }

    // Intrinsics must match the image they're projecting.
    if (info.width() != width || info.height() != height)
    {
      std::cout << "CameraInfo (" << info.width() << "x" << info.height()
                << ") does not match depth image (" << width << "x" << height
                << "); points will be misprojected\n";
    }
    double fx, fy, cx, cy;
    intrinsics_from_camera_info(info, fx, fy, cx, cy);
    const float fxf = static_cast<float>(fx), fyf = static_cast<float>(fy);
    const float cxf = static_cast<float>(cx), cyf = static_cast<float>(cy);

    // --- Layout ---
    const size_t npts = static_cast<size_t>(width) * height;
    if (organized)
    {
      cloud.height = height;
      cloud.width = width;
    }
    else
    {
      cloud.height = 1;
      cloud.width = static_cast<uint32_t>(npts);
    }
    cloud.is_bigendian = false;

    // --- Fields x,y,z (float32) ---
    cloud.fields.clear();
    uint32_t offset = 0;
    cloud.fields.emplace_back(make_float32_field("x", offset));
    offset += 4;
    cloud.fields.emplace_back(make_float32_field("y", offset));
    offset += 4;
    cloud.fields.emplace_back(make_float32_field("z", offset));
    offset += 4;

    cloud.point_step = offset; // 12
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(static_cast<size_t>(cloud.point_step) * npts); // upper bound

    bool all_finite = true;
    auto write_xyz = [&](size_t dst_index, float X, float Y, float Z)
    {
      uint8_t *dst = cloud.data.data() + dst_index * cloud.point_step;
      std::memcpy(dst + cloud.fields[0].offset, &X, 4);
      std::memcpy(dst + cloud.fields[1].offset, &Y, 4);
      std::memcpy(dst + cloud.fields[2].offset, &Z, 4);
    };

    size_t dst = 0;
    for (uint32_t v = 0; v < height; ++v)
    {
      for (uint32_t u = 0; u < width; ++u)
      {
        float d; // safe unaligned read from protobuf bytes
        std::memcpy(&d, src.data() + (static_cast<size_t>(v) * width + u) * 4, 4);

        if (std::isfinite(d) && d > 0.0f)
        {
          const float Z = d;
          const float X = (static_cast<float>(u) - cxf) * d / fxf;
          const float Y = (static_cast<float>(v) - cyf) * d / fyf;
          write_xyz(organized ? (static_cast<size_t>(v) * cloud.width + u) : dst++, X, Y, Z);
        }
        else
        {
          all_finite = false;
          if (organized)
            write_xyz(static_cast<size_t>(v) * cloud.width + u,
                      invalid_value, invalid_value, invalid_value);
        }
      }
    }

    if (!organized)
    {
      cloud.width = static_cast<uint32_t>(dst);
      cloud.row_step = cloud.point_step * cloud.width;
      cloud.data.resize(static_cast<size_t>(cloud.row_step));
    }
    cloud.is_dense = all_finite;
  }

} // namespace sensor_converters
