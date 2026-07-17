#pragma once

#include "tensorrt_runner.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace rclane {

struct LanePoint {
    float x{};
    float y{};
    float score{};
};

struct Lane {
    int width{};
    int height{};
    std::vector<LanePoint> points;
    double score_sum{};
    int lane_id{};
    std::string role;
    bool ego_boundary{};
    int lateral_rank{};

    double score() const;
};

struct DecoderConfig {
    float step_length{10.0F};
    float segmentation_threshold{0.5F};
    float seed_threshold{0.5F};
    int seed_min_distance{2};
    float score_threshold{0.10F};
    float iou_threshold{0.5F};
    int max_seeds{1024};
    int nms_max_lanes{128};
    float nms_scale{0.25F};
    int lane_width{15};
    int max_output_lanes{4};
    float ego_x{400.0F};
    float ego_min_score_ratio{0.5F};
    int threads{8};
};

struct DecodeStatistics {
    std::size_t foreground_pixels{};
    std::size_t seeds{};
    std::size_t crawled_candidates{};
    std::size_t nms_candidates{};
    std::size_t nms_survivors{};
};

std::vector<float> softmax_foreground(const Tensor& segmentation_logits);

std::vector<Lane> decode(
    const std::vector<float>& segmentation_probability,
    const Tensor& up_arrow,
    const Tensor& down_arrow,
    const Tensor& up_bound,
    const Tensor& down_bound,
    const DecoderConfig& config = {},
    DecodeStatistics* statistics = nullptr
);

std::vector<Lane> decode_outputs(
    const std::unordered_map<std::string, Tensor>& outputs,
    const DecoderConfig& config = {},
    DecodeStatistics* statistics = nullptr
);

void write_lanes_json(
    const std::string& path,
    const std::vector<Lane>& lanes,
    const DecodeStatistics* statistics = nullptr
);

}  // namespace rclane
