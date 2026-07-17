#include "bev.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

rclane::CubicFit make_fit(
    double c0,
    double c1 = 0.015,
    double c2 = -0.00005,
    double c3 = 0.0
) {
    rclane::CubicFit fit;
    fit.coefficients = {c0, c1, c2, c3};
    fit.x_min = 40.0;
    fit.x_max = 120.0;
    fit.rmse = 0.02;
    fit.point_count = 80U;
    fit.inlier_count = 80U;
    fit.valid = true;
    return fit;
}

rclane::BevLane make_lane(int lane_id, double c0, double score) {
    rclane::BevLane lane;
    lane.lane_id = lane_id;
    lane.role = lane_id == 0 ? "left_2"
        : lane_id == 1 ? "ego_left"
        : lane_id == 2 ? "ego_right" : "right_2";
    lane.score = score;
    lane.fit = make_fit(c0);
    lane.fit_accepted = true;
    return lane;
}

double evaluate(const rclane::CubicFit& fit, double x) {
    const auto& c = fit.coefficients;
    return ((c[3] * x + c[2]) * x + c[1]) * x + c[0];
}

void require_ordered(const std::vector<rclane::BevLane>& lanes) {
    for (std::size_t index = 1; index < lanes.size(); ++index) {
        const auto& left = lanes[index - 1U];
        const auto& right = lanes[index];
        const double x_min = std::max(left.fit.x_min, right.fit.x_min);
        const double x_max = std::min(left.fit.x_max, right.fit.x_max);
        if (x_max <= x_min) {
            continue;
        }
        for (int sample = 0; sample <= 20; ++sample) {
            const double x = x_min + (x_max - x_min) * sample / 20.0;
            require(
                evaluate(left.fit, x) > evaluate(right.fit, x),
                "parallel output crossed or reversed lane order"
            );
        }
    }
}

std::vector<rclane::BevLane> healthy_lanes() {
    return {
        make_lane(0, 5.25, 0.90),
        make_lane(1, 1.75, 0.93),
        make_lane(2, -1.75, 0.98),
        make_lane(3, -5.25, 0.91),
    };
}

}  // namespace

int main() {
    using rclane::BevMode;

    require(rclane::parse_bev_mode("raw") == BevMode::Raw,
            "raw parser mismatch");
    require(rclane::parse_bev_mode("triggered") == BevMode::Trigger,
            "trigger parser mismatch");
    require(rclane::parse_bev_mode("always_parallel")
                == BevMode::AlwaysParallel,
            "always-parallel parser mismatch");
    require(rclane::parse_bev_mode("complete4") == BevMode::CompleteFour,
            "complete-four parser mismatch");

    rclane::BevConfig config;
    rclane::BevTopologyReport report;

    auto raw = healthy_lanes();
    const auto raw_coefficients = raw[1].fit.coefficients;
    config.mode = BevMode::Raw;
    rclane::apply_bev_mode(raw, config, &report);
    require(!report.applied, "raw mode unexpectedly repaired topology");
    require(raw[1].fit.coefficients == raw_coefficients,
            "raw mode changed cubic coefficients");
    require(std::none_of(raw.begin(), raw.end(), [](const auto& lane) {
        return lane.parallel_repaired || lane.synthetic;
    }), "raw mode published repaired/synthetic lanes");

    auto healthy_trigger = healthy_lanes();
    config.mode = BevMode::Trigger;
    rclane::apply_bev_mode(healthy_trigger, config, &report);
    require(!report.applied,
            "trigger mode repaired an already healthy topology");
    require(report.trigger_pairs.empty(),
            "healthy topology produced a trigger pair");

    auto collapsed = healthy_lanes();
    collapsed[3].fit = make_fit(-1.60, 0.015, -0.00005);
    config.mode = BevMode::Trigger;
    rclane::apply_bev_mode(collapsed, config, &report);
    require(report.applied, "trigger mode did not repair collapsed lanes");
    require(!report.trigger_pairs.empty(),
            "collapsed topology did not report its trigger");
    require(report.reference_lane == 2,
            "trigger mode did not select highest-confidence covered lane");
    require_ordered(collapsed);

    auto forced = healthy_lanes();
    config.mode = BevMode::AlwaysParallel;
    rclane::apply_bev_mode(forced, config, &report);
    require(report.applied && report.forced,
            "always-parallel mode did not force repair");
    require(std::all_of(forced.begin(), forced.end(), [](const auto& lane) {
        return lane.parallel_repaired && !lane.synthetic;
    }), "always-parallel mode did not rebuild every detected lane");
    require_ordered(forced);

    auto complete = healthy_lanes();
    complete.erase(complete.begin());
    complete.pop_back();
    config.mode = BevMode::CompleteFour;
    rclane::apply_bev_mode(complete, config, &report);
    require(report.applied && report.forced && report.funnel_bypassed,
            "complete-four mode report mismatch");
    require(complete.size() == 4U,
            "complete-four mode did not output P0-P3");
    for (int lane_id = 0; lane_id < 4; ++lane_id) {
        require(complete[static_cast<std::size_t>(lane_id)].lane_id == lane_id,
                "complete-four output IDs are not P0-P3");
    }
    require(complete[0].synthetic && complete[3].synthetic,
            "complete-four did not mark missing outer lanes synthetic");
    require(!complete[1].synthetic && !complete[2].synthetic,
            "complete-four marked measured ego lanes synthetic");
    require_ordered(complete);

    const std::string topology_json = rclane::bev_topology_json(report);
    require(topology_json.find("\"mode\":\"complete-four\"")
                != std::string::npos,
            "topology JSON omitted selected mode");
    require(topology_json.find("\"synthetic_lane_ids\":[0,3]")
                != std::string::npos,
            "topology JSON omitted synthetic lane IDs");

    std::cout << "OK -- raw, trigger, always-parallel and complete-four BEV modes\n";
    return 0;
}
