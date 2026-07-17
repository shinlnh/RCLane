#pragma once

#include "decoder.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace rclane {

struct GroundPoint {
    double x{};
    double y{};
    double score{};
};

struct CubicFit {
    std::array<double, 4> coefficients{};
    double x_min{};
    double x_max{};
    double rmse{};
    std::size_t point_count{};
    std::size_t inlier_count{};
    bool valid{};
};

struct BevLane {
    int lane_id{};
    std::string role;
    double score{};
    std::vector<GroundPoint> points;
    CubicFit fit;
    bool fit_accepted{};
    bool funnel_clipped{};
};

struct BevConfig {
    double x_min{0.0};
    double x_max{300.0};
    double y_min{-85.0};
    double y_max{85.0};
    double maximum_rmse{0.5};
    double funnel_margin{0.10};
};

std::vector<BevLane> project_lanes_to_bev(
    const std::vector<Lane>& lanes,
    const BevConfig& config = {}
);

void write_bev_json(
    const std::string& path,
    const std::vector<BevLane>& lanes
);

}  // namespace rclane
