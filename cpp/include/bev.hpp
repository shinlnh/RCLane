#pragma once

#include "decoder.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace rclane {

enum class BevMode {
    Raw,
    Trigger,
    AlwaysParallel,
    CompleteFour,
};

const char* bev_mode_name(BevMode mode);
BevMode parse_bev_mode(const std::string& value);

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
    bool parallel_repaired{};
    bool synthetic{};
    bool has_source_fit{};
    CubicFit source_fit;
    int parallel_reference_lane{-1};
    double parallel_offset_m{};
    std::string parallel_repair_method;
};

struct BevTopologyReport {
    BevMode mode{BevMode::Raw};
    bool applied{};
    bool forced{};
    bool validation_passed{true};
    bool funnel_bypassed{};
    int reference_lane{-1};
    double reference_score{};
    double lane_width_m{3.5};
    std::string activation{"disabled"};
    std::string lane_width_source{"nominal"};
    std::string reference_selection;
    std::string method;
    std::string failure;
    std::vector<std::string> trigger_pairs;
    std::vector<int> synthetic_lane_ids;
};

struct BevConfig {
    double x_min{0.0};
    double x_max{300.0};
    double y_min{-85.0};
    double y_max{85.0};
    double maximum_rmse{0.5};
    double funnel_margin{0.10};
    BevMode mode{BevMode::Raw};
    double nominal_lane_width{3.5};
    double trigger_gap_ratio{0.55};
    double minimum_gap{1.0};
    double minimum_bad_run{4.0};
    double sample_step{0.5};
    double maximum_reference_extrapolation{2.0};
};

std::vector<BevLane> project_lanes_to_bev(
    const std::vector<Lane>& lanes,
    const BevConfig& config = {},
    BevTopologyReport* topology = nullptr
);

void apply_bev_mode(
    std::vector<BevLane>& lanes,
    const BevConfig& config,
    BevTopologyReport* topology = nullptr
);

std::string bev_topology_json(const BevTopologyReport& topology);

void write_bev_json(
    const std::string& path,
    const std::vector<BevLane>& lanes,
    const BevTopologyReport* topology = nullptr
);

}  // namespace rclane
