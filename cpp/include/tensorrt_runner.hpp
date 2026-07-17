#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rclane {

struct Tensor {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};

struct InferenceTiming {
    double mean_ms{};
    double median_ms{};
    double p95_ms{};
    double min_ms{};
    double max_ms{};
};

class TensorRTRunner {
public:
    explicit TensorRTRunner(const std::string& engine_path);
    ~TensorRTRunner();

    TensorRTRunner(const TensorRTRunner&) = delete;
    TensorRTRunner& operator=(const TensorRTRunner&) = delete;
    TensorRTRunner(TensorRTRunner&&) noexcept;
    TensorRTRunner& operator=(TensorRTRunner&&) noexcept;

    const std::vector<std::int64_t>& input_shape() const;
    std::size_t input_elements() const;
    std::unordered_map<std::string, Tensor> infer(const float* input);
    const std::unordered_map<std::string, Tensor>& infer_reuse(
        const float* input
    );
    InferenceTiming benchmark(
        const float* input, int warmup, int iterations
    );

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rclane
