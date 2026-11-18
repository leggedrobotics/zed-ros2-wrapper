// Copyright 2026 Stereolabs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sl_cuda_buffer.hpp"

#ifdef FOUND_CUDA_BUFFER

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <cuda_buffer/cuda_buffer_api.hpp>
#include <rosidl_buffer_backend_registry/buffer_backend_registry.hpp>

namespace sl_tools
{

namespace
{
std::string cudaErrorStr(cudaError_t err)
{
  return std::string(cudaGetErrorName(err)) + ": " + cudaGetErrorString(err);
}

std::string cuErrorStr(CUresult res)
{
  const char * name = nullptr;
  cuGetErrorName(res, &name);
  return name ? std::string(name) : std::to_string(static_cast<int>(res));
}

bool matTypeToEncoding(sl::MAT_TYPE type, std::string & encoding)
{
  switch (type) {
    case sl::MAT_TYPE::F32_C1:
      encoding = sensor_msgs::image_encodings::TYPE_32FC1;
      return true;
    case sl::MAT_TYPE::F32_C2:
      encoding = sensor_msgs::image_encodings::TYPE_32FC2;
      return true;
    case sl::MAT_TYPE::F32_C3:
      encoding = sensor_msgs::image_encodings::TYPE_32FC3;
      return true;
    case sl::MAT_TYPE::F32_C4:
      encoding = sensor_msgs::image_encodings::TYPE_32FC4;
      return true;
    case sl::MAT_TYPE::U8_C1:
      encoding = sensor_msgs::image_encodings::MONO8;
      return true;
    case sl::MAT_TYPE::U8_C2:
      encoding = sensor_msgs::image_encodings::TYPE_8UC2;
      return true;
    case sl::MAT_TYPE::U8_C3:
      encoding = sensor_msgs::image_encodings::BGR8;
      return true;
    case sl::MAT_TYPE::U8_C4:
      encoding = sensor_msgs::image_encodings::BGRA8;
      return true;
    default:
      return false;
  }
}

bool isGpuMat(const sl::Mat & mat, std::string & err)
{
  if (!mat.isInit() || mat.getPtr<sl::uchar1>(sl::MEM::GPU) == nullptr) {
    err = "the sl::Mat has no GPU memory";
    return false;
  }
  return true;
}

void fillImageInfo(
  const sl::Mat & img, uint32_t width, sensor_msgs::msg::Image & msg,
  const std::string & encoding)
{
  int num = 1;  // for endianness detection
  msg.is_bigendian = !(*reinterpret_cast<char *>(&num) == 1);
  msg.height = img.getHeight();
  msg.width = width;
  msg.step = width * img.getPixelBytes();
  msg.encoding = encoding;
}

// Read the version of an installed ROS 2 package from its `package.xml` file
bool getPackageVersion(const std::string & pkg, int & major, int & minor, int & patch)
{
  try {
    std::ifstream file(ament_index_cpp::get_package_share_directory(pkg) + "/package.xml");
    const std::string xml(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto pos = xml.find("<version>");
    return pos != std::string::npos &&
           std::sscanf(xml.c_str() + pos + 9, "%d.%d.%d", &major, &minor, &patch) == 3;
  } catch (const std::exception &) {
    return false;
  }
}
}  // namespace

bool isCudaBufferRmwSupported(const std::string & rmw, std::string & reason)
{
  if (rmw == "rmw_fastrtps_cpp") {
    return true;
  }
  if (rmw == "rmw_zenoh_cpp") {
    // `rosidl::Buffer` backends are supported by rmw_zenoh_cpp v0.12.0 and newer
    // (ros2/rmw_zenoh#930), and by v0.10.5 and newer on Lyrical (backport #987).
    // The v0.11 releases of Rolling do not support them.
    int major = 0, minor = 0, patch = 0;
    if (!getPackageVersion(rmw, major, minor, patch)) {
      reason = "cannot read the version of 'rmw_zenoh_cpp', v0.10.5 or newer is required";
      return false;
    }
    const auto version = std::make_tuple(major, minor, patch);
    const bool supported = (version >= std::make_tuple(0, 10, 5) &&
      version < std::make_tuple(0, 11, 0)) || version >= std::make_tuple(0, 12, 0);
    if (!supported) {
      reason = "'rmw_zenoh_cpp' v" + std::to_string(major) + "." + std::to_string(minor) + "." +
        std::to_string(patch) +
        " does not support it, v0.10.5 (Lyrical) or v0.12.0 or newer is required";
      return false;
    }
    return true;
  }
  reason = "'" + rmw +
    "' does not support it, requires 'rmw_fastrtps_cpp' or 'rmw_zenoh_cpp' v0.10.5 or newer";
  return false;
}

bool isCudaBufferBackendInstalled(std::vector<std::string> & installed)
{
  installed.clear();
  try {
    rosidl_buffer_backend_registry::BufferBackendRegistry registry;
    installed = registry.get_backend_names();
  } catch (const std::exception &) {
    return false;
  }
  return std::find(installed.begin(), installed.end(), CUDA_BUFFER_BACKEND) != installed.end();
}

CudaBufferTransport::~CudaBufferTransport()
{
  reset();
}

bool CudaBufferTransport::init(
  CUcontext sdk_ctx, CUstream sdk_stream, bool retrieve_on_stream,
  std::string & err)
{
  reset();

  std::lock_guard<std::mutex> lock(mMutex);

  if (!sdk_ctx) {
    if (!sdk_stream) {
      err = "neither the CUDA context nor the CUDA stream of the ZED SDK are available";
      return false;
    }
    CUresult res = cuStreamGetCtx(sdk_stream, &sdk_ctx);
    if (res != CUDA_SUCCESS) {
      err = "cannot get the CUDA context of the ZED SDK stream (" + cuErrorStr(res) + ")";
      return false;
    }
  }

  CUresult res = cuCtxSetCurrent(sdk_ctx);
  if (res != CUDA_SUCCESS) {
    err = "cannot set the CUDA context of the ZED SDK (" + cuErrorStr(res) + ")";
    return false;
  }

  // With `retrieve_on_stream` the stream must not synchronize with the legacy
  // default stream, otherwise the copies would wait for the GPU work that the
  // SDK queues after the retrieve. Without it, a blocking stream also waits
  // for the retrieve work that the SDK queues on the legacy default stream.
  cudaError_t cerr = cudaStreamCreateWithFlags(
    &mStream, retrieve_on_stream ? cudaStreamNonBlocking : cudaStreamDefault);
  if (cerr != cudaSuccess) {
    err = "cannot create the CUDA stream (" + cudaErrorStr(cerr) + ")";
    mStream = nullptr;
    return false;
  }

  if (sdk_stream && !retrieve_on_stream) {
    cerr = cudaEventCreateWithFlags(&mSdkEvent, cudaEventDisableTiming);
    if (cerr != cudaSuccess) {
      err = "cannot create the CUDA event (" + cudaErrorStr(cerr) + ")";
      cudaStreamDestroy(mStream);
      mStream = nullptr;
      mSdkEvent = nullptr;
      return false;
    }
  }

  mSdkCtx = sdk_ctx;
  mSdkStream = sdk_stream;
  return true;
}

void CudaBufferTransport::reset()
{
  std::lock_guard<std::mutex> lock(mMutex);

  if (!mStream) {
    return;
  }

  cuCtxSetCurrent(mSdkCtx);
  cudaStreamSynchronize(mStream);
  if (mSdkEvent) {
    cudaEventDestroy(mSdkEvent);
    mSdkEvent = nullptr;
  }
  cudaStreamDestroy(mStream);
  mStream = nullptr;
  mSdkStream = nullptr;
  mSdkCtx = nullptr;
}

bool CudaBufferTransport::isReady() const
{
  std::lock_guard<std::mutex> lock(mMutex);
  return mStream != nullptr;
}

cudaStream_t CudaBufferTransport::stream() const
{
  std::lock_guard<std::mutex> lock(mMutex);
  return mStream;
}

bool CudaBufferTransport::prepare(std::string & err)
{
  if (!mStream) {
    err = "CUDA buffer transport not initialized";
    return false;
  }

  CUresult res = cuCtxSetCurrent(mSdkCtx);
  if (res != CUDA_SUCCESS) {
    err = "cannot set the CUDA context of the ZED SDK (" + cuErrorStr(res) + ")";
    return false;
  }

  if (mSdkEvent) {
    cudaError_t cerr = cudaEventRecord(mSdkEvent, reinterpret_cast<cudaStream_t>(mSdkStream));
    if (cerr == cudaSuccess) {
      cerr = cudaStreamWaitEvent(mStream, mSdkEvent, 0);
    }
    if (cerr != cudaSuccess) {
      err = "cannot synchronize with the ZED SDK stream (" + cudaErrorStr(cerr) + ")";
      return false;
    }
  }
  return true;
}

bool CudaBufferTransport::fillBuffer(
  rosidl::Buffer<uint8_t> & data, size_t size, size_t dst_pitch,
  const std::vector<Copy2D> & copies, std::string & err)
{
  if (size == 0) {
    err = "empty data";
    return false;
  }

  try {
    rosidl::Buffer<uint8_t> buffer = cuda_buffer_backend::allocate_buffer(size);
    cudaError_t cerr = cudaSuccess;
    {
      // The write handle records the write event that the subscribers wait
      // for, when it is destroyed
      cuda_buffer_backend::WriteHandle wh =
        cuda_buffer_backend::from_output_buffer(buffer, mStream);
      for (const auto & c : copies) {
        cerr = cudaMemcpy2DAsync(
          wh.get_ptr() + c.dst_offset, dst_pitch, c.src, c.src_pitch,
          c.width_bytes, c.height, cudaMemcpyDeviceToDevice, mStream);
        if (cerr != cudaSuccess) {
          break;
        }
      }
    }
    // Wait for the end of the copies, so the ZED SDK can safely overwrite the
    // source memory with the next retrieve
    cudaError_t sync_err = cudaStreamSynchronize(mStream);
    if (cerr == cudaSuccess) {
      cerr = sync_err;
    }
    if (cerr != cudaSuccess) {
      err = "GPU copy failed (" + cudaErrorStr(cerr) + ")";
      return false;
    }
    data = std::move(buffer);
  } catch (const std::exception & e) {
    err = std::string("CUDA buffer error: ") + e.what();
    return false;
  }
  return true;
}

bool CudaBufferTransport::toImageMsg(
  const sl::Mat & img, sensor_msgs::msg::Image & msg, std::string & err)
{
  if (!isGpuMat(img, err)) {
    return false;
  }
  std::string encoding;
  if (!matTypeToEncoding(img.getDataType(), encoding)) {
    err = "unsupported sl::MAT_TYPE " + std::to_string(static_cast<int>(img.getDataType()));
    return false;
  }

  std::lock_guard<std::mutex> lock(mMutex);
  if (!prepare(err)) {
    return false;
  }

  sensor_msgs::msg::Image tmp;
  fillImageInfo(img, img.getWidth(), tmp, encoding);
  const Copy2D copy{
    img.getPtr<sl::uchar1>(sl::MEM::GPU), img.getStepBytes(sl::MEM::GPU), 0,
    img.getWidthBytes(), img.getHeight()};
  if (!fillBuffer(tmp.data, tmp.step * tmp.height, tmp.step, {copy}, err)) {
    return false;
  }

  msg.height = tmp.height;
  msg.width = tmp.width;
  msg.encoding = tmp.encoding;
  msg.is_bigendian = tmp.is_bigendian;
  msg.step = tmp.step;
  msg.data = std::move(tmp.data);
  return true;
}

bool CudaBufferTransport::toStereoImageMsg(
  const sl::Mat & left, const sl::Mat & right,
  sensor_msgs::msg::Image & msg, std::string & err)
{
  if (!isGpuMat(left, err) || !isGpuMat(right, err)) {
    return false;
  }
  if (left.getWidth() != right.getWidth() || left.getHeight() != right.getHeight() ||
    left.getDataType() != right.getDataType())
  {
    err = "left and right images have different sizes or types";
    return false;
  }
  std::string encoding;
  if (!matTypeToEncoding(left.getDataType(), encoding)) {
    err = "unsupported sl::MAT_TYPE " + std::to_string(static_cast<int>(left.getDataType()));
    return false;
  }

  std::lock_guard<std::mutex> lock(mMutex);
  if (!prepare(err)) {
    return false;
  }

  sensor_msgs::msg::Image tmp;
  fillImageInfo(left, 2 * left.getWidth(), tmp, encoding);
  const size_t width_bytes = left.getWidthBytes();
  const std::vector<Copy2D> copies{
    {left.getPtr<sl::uchar1>(sl::MEM::GPU), left.getStepBytes(sl::MEM::GPU), 0,
      width_bytes, left.getHeight()},
    {right.getPtr<sl::uchar1>(sl::MEM::GPU), right.getStepBytes(sl::MEM::GPU), width_bytes,
      width_bytes, right.getHeight()}};
  if (!fillBuffer(tmp.data, tmp.step * tmp.height, tmp.step, copies, err)) {
    return false;
  }

  msg.height = tmp.height;
  msg.width = tmp.width;
  msg.encoding = tmp.encoding;
  msg.is_bigendian = tmp.is_bigendian;
  msg.step = tmp.step;
  msg.data = std::move(tmp.data);
  return true;
}

bool CudaBufferTransport::toPointCloudData(
  const sl::Mat & cloud, sensor_msgs::msg::PointCloud2 & msg, std::string & err)
{
  if (!isGpuMat(cloud, err)) {
    return false;
  }
  if (cloud.getDataType() != sl::MAT_TYPE::F32_C4) {
    err = "the point cloud is not a F32_C4 sl::Mat";
    return false;
  }

  std::lock_guard<std::mutex> lock(mMutex);
  if (!prepare(err)) {
    return false;
  }

  const size_t width_bytes = cloud.getWidthBytes();
  const Copy2D copy{
    cloud.getPtr<sl::uchar1>(sl::MEM::GPU), cloud.getStepBytes(sl::MEM::GPU), 0,
    width_bytes, cloud.getHeight()};
  return fillBuffer(msg.data, width_bytes * cloud.getHeight(), width_bytes, {copy}, err);
}

bool CudaBufferTransport::download(const sl::Mat & gpu, sl::Mat & cpu, std::string & err)
{
  if (!isGpuMat(gpu, err)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mMutex);
  if (!prepare(err)) {
    return false;
  }

  sl::ERROR_CODE sl_err = gpu.copyTo(cpu, sl::COPY_TYPE::GPU_CPU, mStream);
  cudaError_t cerr = cudaStreamSynchronize(mStream);
  if (sl_err != sl::ERROR_CODE::SUCCESS) {
    err = std::string("GPU to CPU copy failed (") + sl::toString(sl_err).c_str() + ")";
    return false;
  }
  if (cerr != cudaSuccess) {
    err = "GPU to CPU copy failed (" + cudaErrorStr(cerr) + ")";
    return false;
  }
  return true;
}

bool CudaBufferTransport::download(
  const sl::Mat & gpu, void * dst, size_t dst_pitch, std::string & err)
{
  if (!isGpuMat(gpu, err)) {
    return false;
  }
  if (!dst || dst_pitch < gpu.getWidthBytes()) {
    err = "invalid destination buffer";
    return false;
  }

  std::lock_guard<std::mutex> lock(mMutex);
  if (!prepare(err)) {
    return false;
  }

  cudaError_t cerr = cudaMemcpy2DAsync(
    dst, dst_pitch, gpu.getPtr<sl::uchar1>(sl::MEM::GPU), gpu.getStepBytes(sl::MEM::GPU),
    gpu.getWidthBytes(), gpu.getHeight(), cudaMemcpyDeviceToHost, mStream);
  cudaError_t sync_err = cudaStreamSynchronize(mStream);
  if (cerr == cudaSuccess) {
    cerr = sync_err;
  }
  if (cerr != cudaSuccess) {
    err = "GPU to CPU copy failed (" + cudaErrorStr(cerr) + ")";
    return false;
  }
  return true;
}

}  // namespace sl_tools

#endif  // FOUND_CUDA_BUFFER
