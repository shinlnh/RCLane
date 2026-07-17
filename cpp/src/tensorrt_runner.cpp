#include "tensorrt_runner.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace rclane {
namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            // TensorRT owns message storage; copying is unnecessary here.
            last_message_ = message == nullptr ? "" : message;
        }
    }

    const std::string& last_message() const { return last_message_; }

private:
    std::string last_message_;
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

std::vector<char> read_binary(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open TensorRT engine: " + path);
    }
    const auto end = stream.tellg();
    if (end <= 0) {
        throw std::runtime_error("empty TensorRT engine: " + path);
    }
    std::vector<char> bytes(static_cast<std::size_t>(end));
    stream.seekg(0);
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("cannot read TensorRT engine: " + path);
    }
    return bytes;
}

std::vector<std::int64_t> dimensions(const nvinfer1::Dims& dims) {
    std::vector<std::int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::runtime_error("dynamic/invalid TensorRT shape");
        }
        shape.push_back(dims.d[index]);
    }
    return shape;
}

std::size_t element_count(const std::vector<std::int64_t>& shape) {
    return std::accumulate(
        shape.begin(), shape.end(), std::size_t{1},
        [](std::size_t total, std::int64_t value) {
            return total * static_cast<std::size_t>(value);
        }
    );
}

struct DeviceTensor {
    std::string name;
    bool input{};
    std::vector<std::int64_t> shape;
    std::size_t elements{};
    void* device{};
};

}  // namespace

class TensorRTRunner::Impl {
public:
    explicit Impl(const std::string& engine_path) {
        const auto bytes = read_binary(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("createInferRuntime failed");
        }
        engine_.reset(runtime_->deserializeCudaEngine(
            bytes.data(), bytes.size()
        ));
        if (!engine_) {
            throw std::runtime_error(
                "deserializeCudaEngine failed: " + logger_.last_message()
            );
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("createExecutionContext failed");
        }
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

        const int tensor_count = engine_->getNbIOTensors();
        for (int index = 0; index < tensor_count; ++index) {
            const char* tensor_name = engine_->getIOTensorName(index);
            if (engine_->getTensorDataType(tensor_name)
                != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error(
                    std::string("non-FP32 engine tensor: ") + tensor_name
                );
            }
            DeviceTensor tensor;
            tensor.name = tensor_name;
            tensor.input = engine_->getTensorIOMode(tensor_name)
                == nvinfer1::TensorIOMode::kINPUT;
            tensor.shape = dimensions(engine_->getTensorShape(tensor_name));
            tensor.elements = element_count(tensor.shape);
            check_cuda(
                cudaMalloc(&tensor.device, tensor.elements * sizeof(float)),
                "cudaMalloc"
            );
            if (!context_->setTensorAddress(tensor.name.c_str(), tensor.device)) {
                throw std::runtime_error(
                    "setTensorAddress failed for " + tensor.name
                );
            }
            if (tensor.input) {
                if (input_index_ >= 0) {
                    throw std::runtime_error("engine has multiple inputs");
                }
                input_index_ = static_cast<int>(tensors_.size());
            }
            tensors_.push_back(std::move(tensor));
        }
        if (input_index_ < 0) {
            throw std::runtime_error("engine has no input tensor");
        }
    }

    ~Impl() {
        for (auto& tensor : tensors_) {
            if (tensor.device != nullptr) {
                cudaFree(tensor.device);
            }
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    void enqueue(const float* input) {
        const auto& tensor = tensors_[static_cast<std::size_t>(input_index_)];
        check_cuda(cudaMemcpyAsync(
            tensor.device, input, tensor.elements * sizeof(float),
            cudaMemcpyHostToDevice, stream_
        ), "cudaMemcpyAsync input");
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }
    }

    void synchronize() {
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

    std::unordered_map<std::string, Tensor> infer(const float* input) {
        enqueue(input);
        std::unordered_map<std::string, Tensor> output;
        for (const auto& tensor : tensors_) {
            if (tensor.input) {
                continue;
            }
            Tensor host;
            host.shape = tensor.shape;
            host.values.resize(tensor.elements);
            check_cuda(cudaMemcpyAsync(
                host.values.data(), tensor.device,
                tensor.elements * sizeof(float), cudaMemcpyDeviceToHost,
                stream_
            ), "cudaMemcpyAsync output");
            output.emplace(tensor.name, std::move(host));
        }
        synchronize();
        return output;
    }

    const std::unordered_map<std::string, Tensor>& infer_reuse(
        const float* input
    ) {
        if (host_outputs_.empty()) {
            for (const auto& tensor : tensors_) {
                if (tensor.input) {
                    continue;
                }
                Tensor host;
                host.shape = tensor.shape;
                host.values.resize(tensor.elements);
                host_outputs_.emplace(tensor.name, std::move(host));
            }
        }
        enqueue(input);
        for (const auto& tensor : tensors_) {
            if (tensor.input) {
                continue;
            }
            auto& host = host_outputs_.at(tensor.name);
            check_cuda(cudaMemcpyAsync(
                host.values.data(), tensor.device,
                tensor.elements * sizeof(float), cudaMemcpyDeviceToHost,
                stream_
            ), "cudaMemcpyAsync reusable output");
        }
        synchronize();
        return host_outputs_;
    }

    InferenceTiming benchmark(const float* input, int warmup, int iterations) {
        if (warmup < 0 || iterations <= 0) {
            throw std::invalid_argument("invalid benchmark iteration count");
        }
        for (int index = 0; index < warmup; ++index) {
            enqueue(input);
            synchronize();
        }
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(iterations));
        for (int index = 0; index < iterations; ++index) {
            const auto started = std::chrono::steady_clock::now();
            enqueue(input);
            synchronize();
            const auto stopped = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(
                stopped - started
            ).count());
        }
        std::sort(samples.begin(), samples.end());
        const auto percentile = [&samples](double fraction) {
            const auto position = static_cast<std::size_t>(std::floor(
                fraction * static_cast<double>(samples.size() - 1)
            ));
            return samples[position];
        };
        const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        return {
            sum / static_cast<double>(samples.size()),
            percentile(0.50),
            percentile(0.95),
            samples.front(),
            samples.back(),
        };
    }

    const std::vector<std::int64_t>& input_shape() const {
        return tensors_[static_cast<std::size_t>(input_index_)].shape;
    }

    std::size_t input_elements() const {
        return tensors_[static_cast<std::size_t>(input_index_)].elements;
    }

private:
    Logger logger_;
    struct RuntimeDeleter {
        template <typename T>
        void operator()(T* pointer) const { delete pointer; }
    };
    std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, RuntimeDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, RuntimeDeleter> context_;
    cudaStream_t stream_{};
    std::vector<DeviceTensor> tensors_;
    std::unordered_map<std::string, Tensor> host_outputs_;
    int input_index_{-1};
};

TensorRTRunner::TensorRTRunner(const std::string& engine_path)
    : impl_(std::make_unique<Impl>(engine_path)) {}
TensorRTRunner::~TensorRTRunner() = default;
TensorRTRunner::TensorRTRunner(TensorRTRunner&&) noexcept = default;
TensorRTRunner& TensorRTRunner::operator=(TensorRTRunner&&) noexcept = default;

const std::vector<std::int64_t>& TensorRTRunner::input_shape() const {
    return impl_->input_shape();
}
std::size_t TensorRTRunner::input_elements() const {
    return impl_->input_elements();
}
std::unordered_map<std::string, Tensor> TensorRTRunner::infer(
    const float* input
) {
    return impl_->infer(input);
}
const std::unordered_map<std::string, Tensor>& TensorRTRunner::infer_reuse(
    const float* input
) {
    return impl_->infer_reuse(input);
}
InferenceTiming TensorRTRunner::benchmark(
    const float* input, int warmup, int iterations
) {
    return impl_->benchmark(input, warmup, iterations);
}

}  // namespace rclane
