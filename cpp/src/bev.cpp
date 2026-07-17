#include "bev.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace rclane {
namespace {

constexpr double kFx = 3582.768775;
constexpr double kFy = 3582.768775;
constexpr double kCx = 960.0;
constexpr double kCy = 540.0;
constexpr double kRawWidth = 1920.0;
constexpr double kRawHeight = 1080.0;
constexpr double kModelWidth = 800.0;
constexpr double kModelHeight = 320.0;
constexpr double kCameraX = 1.0;
constexpr double kCameraZ = 1.8;
constexpr double kCosPitch = 0.997564;
constexpr double kSinDown = 0.069756;
constexpr double kPi = 3.14159265358979323846;

bool solve_four_by_four(
    std::array<std::array<double, 4>, 4> matrix,
    std::array<double, 4> target,
    std::array<double, 4>& solution
) {
    for (int column = 0; column < 4; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 4; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(matrix[pivot][column]) < 1e-12) {
            return false;
        }
        std::swap(matrix[pivot], matrix[column]);
        std::swap(target[pivot], target[column]);
        const double divisor = matrix[column][column];
        for (int index = column; index < 4; ++index) {
            matrix[column][index] /= divisor;
        }
        target[column] /= divisor;
        for (int row = 0; row < 4; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = matrix[row][column];
            for (int index = column; index < 4; ++index) {
                matrix[row][index] -= factor * matrix[column][index];
            }
            target[row] -= factor * target[column];
        }
    }
    solution = target;
    return true;
}

bool weighted_fit(
    const std::vector<double>& z,
    const std::vector<double>& y,
    const std::vector<double>& weights,
    const std::vector<std::uint8_t>* include,
    std::array<double, 4>& coefficients
) {
    std::array<std::array<double, 4>, 4> normal{};
    std::array<double, 4> target{};
    for (std::size_t index = 0; index < z.size(); ++index) {
        if (include != nullptr && (*include)[index] == 0U) {
            continue;
        }
        const std::array<double, 4> row{1.0, z[index], z[index] * z[index],
                                        z[index] * z[index] * z[index]};
        for (int i = 0; i < 4; ++i) {
            target[i] += weights[index] * row[i] * y[index];
            for (int j = 0; j < 4; ++j) {
                normal[i][j] += weights[index] * row[i] * row[j];
            }
        }
    }
    return solve_four_by_four(normal, target, coefficients);
}

double evaluate(const std::array<double, 4>& c, double x) {
    return ((c[3] * x + c[2]) * x + c[1]) * x + c[0];
}

std::vector<GroundPoint> project_lane(
    const Lane& lane, const BevConfig& config
) {
    std::vector<GroundPoint> ground;
    ground.reserve(lane.points.size());
    for (const auto& point : lane.points) {
        const double u = point.x * (kRawWidth / kModelWidth);
        const double v = point.y * (kRawHeight / kModelHeight);
        const double ray_right = (u - kCx) / kFx;
        const double ray_down = (v - kCy) / kFy;
        // OpenCV optical -> UE camera is (forward,right,up)=(1,x,-y),
        // followed by the fixed camera-to-vehicle pitch rotation.
        const double ray_x = kCosPitch - kSinDown * ray_down;
        const double ray_y = ray_right;
        const double ray_z = -kSinDown - kCosPitch * ray_down;
        if (!(ray_z < -1e-8)) {
            continue;
        }
        const double scale = -kCameraZ / ray_z;
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            continue;
        }
        const double x = kCameraX + scale * ray_x;
        const double y_left = -scale * ray_y;
        if (std::isfinite(x) && std::isfinite(y_left)
            && x >= config.x_min && x <= config.x_max
            && y_left >= config.y_min && y_left <= config.y_max) {
            ground.push_back({x, y_left, point.score});
        }
    }
    return ground;
}

CubicFit fit_cubic(const std::vector<GroundPoint>& input) {
    CubicFit fit;
    fit.point_count = input.size();
    if (input.size() < 6) {
        return fit;
    }
    std::vector<GroundPoint> points = input;
    std::sort(points.begin(), points.end(), [](const GroundPoint& lhs,
                                               const GroundPoint& rhs) {
        return lhs.x < rhs.x;
    });
    const double minimum_x = points.front().x;
    const double maximum_x = points.back().x;
    if (maximum_x - minimum_x < 1.0) {
        return fit;
    }
    const double center = (minimum_x + maximum_x) * 0.5;
    const double scale = std::max((maximum_x - minimum_x) * 0.5, 1.0);
    std::vector<double> z(points.size());
    std::vector<double> y(points.size());
    std::vector<double> base(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        z[index] = (points[index].x - center) / scale;
        y[index] = points[index].y;
        base[index] = std::clamp(points[index].score, 0.05, 1.0);
    }
    std::vector<double> weights = base;
    std::array<double, 4> normalized{};
    if (!weighted_fit(z, y, weights, nullptr, normalized)) {
        return fit;
    }
    for (int iteration = 0; iteration < 3; ++iteration) {
        for (std::size_t index = 0; index < points.size(); ++index) {
            const double residual = y[index] - evaluate(normalized, z[index]);
            const double robust = std::min(
                1.0, 0.30 / std::max(std::abs(residual), 1e-8)
            );
            weights[index] = base[index] * robust;
        }
        if (!weighted_fit(z, y, weights, nullptr, normalized)) {
            return fit;
        }
    }
    std::vector<double> absolute_residual(points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        absolute_residual[index] = std::abs(
            y[index] - evaluate(normalized, z[index])
        );
    }
    std::vector<double> sorted_residual = absolute_residual;
    std::sort(sorted_residual.begin(), sorted_residual.end());
    const std::size_t middle = sorted_residual.size() / 2;
    const double median_residual = (sorted_residual.size() & 1U) != 0U
        ? sorted_residual[middle]
        : (sorted_residual[middle - 1] + sorted_residual[middle]) * 0.5;
    const double threshold = std::max(0.30, 2.5 * median_residual);
    std::vector<std::uint8_t> inliers(points.size(), 0U);
    std::size_t inlier_count = 0;
    for (std::size_t index = 0; index < points.size(); ++index) {
        inliers[index] = static_cast<std::uint8_t>(
            absolute_residual[index] <= threshold
        );
        inlier_count += inliers[index];
    }
    if (inlier_count >= 6) {
        if (!weighted_fit(z, y, base, &inliers, normalized)) {
            return fit;
        }
    } else {
        std::fill(inliers.begin(), inliers.end(), 1U);
        inlier_count = points.size();
    }

    const double inverse_scale = 1.0 / scale;
    const double argument_constant = -center * inverse_scale;
    const double argument_linear = inverse_scale;
    fit.coefficients[0] = normalized[0]
        + normalized[1] * argument_constant
        + normalized[2] * argument_constant * argument_constant
        + normalized[3] * argument_constant * argument_constant * argument_constant;
    fit.coefficients[1] = normalized[1] * argument_linear
        + 2.0 * normalized[2] * argument_constant * argument_linear
        + 3.0 * normalized[3] * argument_constant * argument_constant * argument_linear;
    fit.coefficients[2] = normalized[2] * argument_linear * argument_linear
        + 3.0 * normalized[3] * argument_constant * argument_linear * argument_linear;
    fit.coefficients[3] = normalized[3] * argument_linear * argument_linear * argument_linear;

    fit.x_min = std::numeric_limits<double>::infinity();
    fit.x_max = -std::numeric_limits<double>::infinity();
    double square_error = 0.0;
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (inliers[index] == 0U) {
            continue;
        }
        fit.x_min = std::min(fit.x_min, points[index].x);
        fit.x_max = std::max(fit.x_max, points[index].x);
        const double residual = points[index].y
            - evaluate(fit.coefficients, points[index].x);
        square_error += residual * residual;
    }
    fit.inlier_count = inlier_count;
    fit.rmse = std::sqrt(square_error / static_cast<double>(inlier_count));
    fit.valid = true;
    return fit;
}

double evaluate_derivative(const std::array<double, 4>& c, double x) {
    return (3.0 * c[3] * x + 2.0 * c[2]) * x + c[1];
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    return (values.size() & 1U) != 0U
        ? values[middle]
        : (values[middle - 1U] + values[middle]) * 0.5;
}

std::vector<double> shared_samples(
    const CubicFit& left,
    const CubicFit& right,
    double sample_step
) {
    const double x_min = std::max(left.x_min, right.x_min);
    const double x_max = std::min(left.x_max, right.x_max);
    if (x_max - x_min < sample_step) {
        return {};
    }
    const int count = std::max(
        3, static_cast<int>(std::ceil((x_max - x_min) / sample_step)) + 1
    );
    std::vector<double> result(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        result[static_cast<std::size_t>(index)] = x_min
            + (x_max - x_min) * static_cast<double>(index)
                / static_cast<double>(count - 1);
    }
    return result;
}

double normal_gap(const CubicFit& left, const CubicFit& right, double x) {
    const double mean_slope = 0.5 * (
        evaluate_derivative(left.coefficients, x)
        + evaluate_derivative(right.coefficients, x)
    );
    return (evaluate(left.coefficients, x) - evaluate(right.coefficients, x))
        / std::sqrt(1.0 + mean_slope * mean_slope);
}

struct FitView {
    int lane_id{};
    double score{};
    const CubicFit* fit{};
};

std::vector<FitView> fit_views(const std::vector<BevLane>& lanes) {
    std::vector<FitView> result;
    for (const BevLane& lane : lanes) {
        if (lane.fit_accepted && lane.fit.valid) {
            result.push_back({lane.lane_id, lane.score, &lane.fit});
        }
    }
    std::sort(result.begin(), result.end(), [](const FitView& lhs,
                                               const FitView& rhs) {
        return lhs.lane_id < rhs.lane_id;
    });
    return result;
}

struct TopologyAnalysis {
    std::vector<std::string> trigger_pairs;
    bool all_positive{true};
};

TopologyAnalysis analyze_topology(
    const std::vector<FitView>& ordered,
    double lane_width,
    const BevConfig& config
) {
    TopologyAnalysis result;
    for (std::size_t index = 1; index < ordered.size(); ++index) {
        const FitView& left = ordered[index - 1U];
        const FitView& right = ordered[index];
        const int lane_steps = right.lane_id - left.lane_id;
        if (lane_steps <= 0) {
            continue;
        }
        const auto x = shared_samples(
            *left.fit, *right.fit, config.sample_step
        );
        if (x.size() < 3U) {
            continue;
        }
        const double expected_gap = lane_width * lane_steps;
        const double trigger_gap = std::max(
            config.minimum_gap, config.trigger_gap_ratio * expected_gap
        );
        bool crossing = false;
        double longest_bad_run = 0.0;
        int bad_start = -1;
        for (std::size_t sample = 0; sample < x.size(); ++sample) {
            const double gap = normal_gap(*left.fit, *right.fit, x[sample]);
            if (!std::isfinite(gap)) {
                if (bad_start >= 0) {
                    const int end = static_cast<int>(sample) - 1;
                    if (end > bad_start) {
                        longest_bad_run = std::max(
                            longest_bad_run,
                            x[static_cast<std::size_t>(end)]
                                - x[static_cast<std::size_t>(bad_start)]
                        );
                    }
                    bad_start = -1;
                }
                continue;
            }
            crossing = crossing || gap <= 0.0;
            result.all_positive = result.all_positive && gap > 0.0;
            const bool bad = gap < trigger_gap;
            if (bad && bad_start < 0) {
                bad_start = static_cast<int>(sample);
            }
            if (bad_start >= 0 && (!bad || sample + 1U == x.size())) {
                const int end = bad && sample + 1U == x.size()
                    ? static_cast<int>(sample)
                    : static_cast<int>(sample) - 1;
                if (end > bad_start) {
                    longest_bad_run = std::max(
                        longest_bad_run,
                        x[static_cast<std::size_t>(end)]
                            - x[static_cast<std::size_t>(bad_start)]
                    );
                }
                bad_start = -1;
            }
        }
        if (crossing || longest_bad_run >= config.minimum_bad_run) {
            result.trigger_pairs.push_back(
                "P" + std::to_string(left.lane_id)
                + "-P" + std::to_string(right.lane_id)
            );
        }
    }
    return result;
}

std::pair<double, std::string> estimate_lane_width(
    const std::vector<FitView>& ordered,
    const BevConfig& config
) {
    std::vector<double> candidates;
    for (std::size_t index = 1; index < ordered.size(); ++index) {
        const FitView& left = ordered[index - 1U];
        const FitView& right = ordered[index];
        const int lane_steps = right.lane_id - left.lane_id;
        if (lane_steps <= 0) {
            continue;
        }
        const auto x = shared_samples(
            *left.fit, *right.fit, config.sample_step
        );
        if (x.size() < 3U) {
            continue;
        }
        const std::size_t near_count = std::max<std::size_t>(
            3U, static_cast<std::size_t>(std::ceil(x.size() * 0.40))
        );
        std::vector<double> plausible;
        for (std::size_t sample = 0;
             sample < std::min(near_count, x.size()); ++sample) {
            const double gap = normal_gap(
                *left.fit, *right.fit, x[sample]
            ) / lane_steps;
            if (std::isfinite(gap) && gap >= 2.4 && gap <= 5.0) {
                plausible.push_back(gap);
            }
        }
        if (!plausible.empty()) {
            candidates.push_back(median(std::move(plausible)));
        }
    }
    if (candidates.empty()) {
        return {config.nominal_lane_width, "nominal"};
    }
    return {
        std::clamp(median(std::move(candidates)), 2.6, 4.5),
        "near_range_median",
    };
}

double shared_domain_support(
    const FitView& candidate,
    const std::vector<FitView>& ordered
) {
    double support = 0.0;
    for (const FitView& other : ordered) {
        if (&other == &candidate) {
            continue;
        }
        support += std::max(
            0.0,
            std::min(candidate.fit->x_max, other.fit->x_max)
                - std::max(candidate.fit->x_min, other.fit->x_min)
        );
    }
    return support;
}

CubicFit parallel_offset_fit(
    const CubicFit& reference,
    const CubicFit& target,
    double offset,
    double sample_step
) {
    const double x_min = std::max(reference.x_min, target.x_min);
    const double x_max = std::min(reference.x_max, target.x_max);
    const double span = x_max - x_min;
    if (span < 1.0) {
        return {};
    }
    const int count = std::max(
        16, static_cast<int>(std::ceil(span / sample_step)) + 1
    );
    std::vector<GroundPoint> points;
    points.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        const double x = x_min + span * static_cast<double>(index)
            / static_cast<double>(count - 1);
        const double y = evaluate(reference.coefficients, x);
        const double slope = evaluate_derivative(reference.coefficients, x);
        const double norm = std::sqrt(1.0 + slope * slope);
        points.push_back({
            x - offset * slope / norm,
            y + offset / norm,
            1.0,
        });
    }
    return fit_cubic(points);
}

std::vector<FitView> views_from_fits(
    const std::vector<FitView>& source,
    const std::map<int, CubicFit>& fits
) {
    std::vector<FitView> result;
    result.reserve(source.size());
    for (const FitView& view : source) {
        const auto found = fits.find(view.lane_id);
        if (found != fits.end()) {
            result.push_back({view.lane_id, view.score, &found->second});
        }
    }
    return result;
}

BevLane* find_lane(std::vector<BevLane>& lanes, int lane_id) {
    const auto found = std::find_if(
        lanes.begin(), lanes.end(), [lane_id](const BevLane& lane) {
            return lane.lane_id == lane_id;
        }
    );
    return found == lanes.end() ? nullptr : &*found;
}

void repair_parallel_lanes(
    std::vector<BevLane>& lanes,
    const BevConfig& config,
    bool force,
    BevTopologyReport& report
) {
    const auto ordered = fit_views(lanes);
    report.forced = force;
    report.activation = force ? "always_parallel" : "triggered";
    if (ordered.size() < 2U) {
        return;
    }
    const auto [lane_width, width_source] = estimate_lane_width(
        ordered, config
    );
    report.lane_width_m = lane_width;
    report.lane_width_source = width_source;
    const auto before = analyze_topology(ordered, lane_width, config);
    report.trigger_pairs = before.trigger_pairs;
    if (before.trigger_pairs.empty() && !force) {
        return;
    }

    const double required_min = std::min_element(
        ordered.begin(), ordered.end(), [](const FitView& lhs,
                                           const FitView& rhs) {
            return lhs.fit->x_min < rhs.fit->x_min;
        }
    )->fit->x_min;
    const double required_max = std::max_element(
        ordered.begin(), ordered.end(), [](const FitView& lhs,
                                           const FitView& rhs) {
            return lhs.fit->x_max < rhs.fit->x_max;
        }
    )->fit->x_max;
    const auto uncovered = [required_min, required_max](const FitView& view) {
        return std::max(
            std::max(0.0, view.fit->x_min - required_min),
            std::max(0.0, required_max - view.fit->x_max)
        );
    };

    const FitView* reference = nullptr;
    if (force) {
        reference = &*std::max_element(
            ordered.begin(), ordered.end(), [&ordered](const FitView& lhs,
                                                       const FitView& rhs) {
                return std::make_tuple(
                    shared_domain_support(lhs, ordered), lhs.score,
                    lhs.fit->x_max - lhs.fit->x_min, -lhs.lane_id
                ) < std::make_tuple(
                    shared_domain_support(rhs, ordered), rhs.score,
                    rhs.fit->x_max - rhs.fit->x_min, -rhs.lane_id
                );
            }
        );
        report.reference_selection = "maximum_shared_domain_then_confidence";
    } else {
        std::vector<const FitView*> coverage;
        for (const FitView& view : ordered) {
            if (uncovered(view) <= config.maximum_reference_extrapolation) {
                coverage.push_back(&view);
            }
        }
        if (!coverage.empty()) {
            reference = *std::max_element(
                coverage.begin(), coverage.end(), [](const FitView* lhs,
                                                      const FitView* rhs) {
                    return std::make_pair(lhs->score, -lhs->lane_id)
                        < std::make_pair(rhs->score, -rhs->lane_id);
                }
            );
            report.reference_selection = "highest_confidence_with_domain_coverage";
        } else {
            reference = &*std::min_element(
                ordered.begin(), ordered.end(), [&uncovered](const FitView& lhs,
                                                              const FitView& rhs) {
                    return std::make_tuple(
                        uncovered(lhs), -lhs.score, lhs.lane_id
                    ) < std::make_tuple(
                        uncovered(rhs), -rhs.score, rhs.lane_id
                    );
                }
            );
            report.reference_selection = "best_domain_coverage_then_confidence";
        }
    }

    std::map<int, double> offsets;
    std::map<int, CubicFit> repaired;
    bool normal_ok = true;
    for (const FitView& view : ordered) {
        const double offset = (
            reference->lane_id - view.lane_id
        ) * lane_width;
        offsets[view.lane_id] = offset;
        if (view.lane_id == reference->lane_id) {
            repaired[view.lane_id] = *reference->fit;
        } else {
            CubicFit fit = parallel_offset_fit(
                *reference->fit, *view.fit, offset, config.sample_step
            );
            if (!fit.valid) {
                normal_ok = false;
                break;
            }
            repaired[view.lane_id] = fit;
        }
    }

    report.method = "normal_offset_cubic";
    TopologyAnalysis after;
    if (normal_ok) {
        after = analyze_topology(
            views_from_fits(ordered, repaired), lane_width, config
        );
    } else {
        after.trigger_pairs.push_back("normal_offset_fit_failed");
    }

    std::set<int> passthrough;
    if (!after.trigger_pairs.empty()) {
        report.method = "shared_shape_vertical_offset_fallback";
        repaired.clear();
        for (const FitView& view : ordered) {
            CubicFit fit = *view.fit;
            fit.x_min = std::max(reference->fit->x_min, view.fit->x_min);
            fit.x_max = std::min(reference->fit->x_max, view.fit->x_max);
            if (fit.x_max - fit.x_min < 1.0) {
                fit = *view.fit;
                passthrough.insert(view.lane_id);
            } else {
                fit.coefficients = reference->fit->coefficients;
                fit.coefficients[0] += offsets[view.lane_id];
                fit.rmse = 0.0;
            }
            repaired[view.lane_id] = fit;
        }
        after = analyze_topology(
            views_from_fits(ordered, repaired), lane_width, config
        );
    }

    report.reference_lane = reference->lane_id;
    report.reference_score = reference->score;
    report.validation_passed = after.trigger_pairs.empty()
        && after.all_positive;
    if (!report.validation_passed) {
        report.failure = "repaired topology did not pass ordering validation";
        return;
    }

    for (const FitView& view : ordered) {
        if (passthrough.count(view.lane_id) != 0U) {
            continue;
        }
        BevLane* lane = find_lane(lanes, view.lane_id);
        if (lane == nullptr) {
            continue;
        }
        lane->source_fit = lane->fit;
        lane->has_source_fit = true;
        lane->fit = repaired.at(view.lane_id);
        lane->parallel_repaired = true;
        lane->parallel_reference_lane = reference->lane_id;
        lane->parallel_offset_m = offsets.at(view.lane_id);
        lane->parallel_repair_method = report.method;
    }
    report.applied = true;
}

std::string role_for_lane(int lane_id) {
    switch (lane_id) {
    case 0: return "left_2";
    case 1: return "ego_left";
    case 2: return "ego_right";
    case 3: return "right_2";
    default: return "lane_" + std::to_string(lane_id);
    }
}

void complete_four_lanes(
    std::vector<BevLane>& lanes,
    const BevConfig& config,
    BevTopologyReport& report
) {
    const auto ordered = fit_views(lanes);
    report.forced = true;
    report.funnel_bypassed = true;
    report.activation = "complete_four_parallel";
    report.reference_selection = "maximum_shared_domain_then_confidence";
    report.validation_passed = false;
    if (ordered.empty()) {
        report.failure = "no valid measured BEV fit is available";
        return;
    }

    double lane_width = config.nominal_lane_width;
    std::string width_source = "nominal_single_reference";
    if (ordered.size() >= 2U) {
        std::tie(lane_width, width_source) = estimate_lane_width(
            ordered, config
        );
        report.trigger_pairs = analyze_topology(
            ordered, lane_width, config
        ).trigger_pairs;
    }
    report.lane_width_m = lane_width;
    report.lane_width_source = width_source;

    const FitView& reference = *std::max_element(
        ordered.begin(), ordered.end(), [&ordered](const FitView& lhs,
                                                   const FitView& rhs) {
            return std::make_tuple(
                shared_domain_support(lhs, ordered), lhs.score,
                lhs.fit->x_max - lhs.fit->x_min, -lhs.lane_id
            ) < std::make_tuple(
                shared_domain_support(rhs, ordered), rhs.score,
                rhs.fit->x_max - rhs.fit->x_min, -rhs.lane_id
            );
        }
    );

    std::map<int, CubicFit> completed;
    std::map<int, double> offsets;
    bool normal_ok = true;
    for (int lane_id = 0; lane_id < 4; ++lane_id) {
        const double offset = (reference.lane_id - lane_id) * lane_width;
        offsets[lane_id] = offset;
        if (lane_id == reference.lane_id) {
            completed[lane_id] = *reference.fit;
        } else {
            CubicFit fit = parallel_offset_fit(
                *reference.fit, *reference.fit, offset, config.sample_step
            );
            if (!fit.valid) {
                normal_ok = false;
                break;
            }
            completed[lane_id] = fit;
        }
    }

    std::vector<FitView> completed_views;
    const auto rebuild_completed_views = [&]() {
        completed_views.clear();
        for (int lane_id = 0; lane_id < 4; ++lane_id) {
            completed_views.push_back({
                lane_id, 1.0, &completed.at(lane_id)
            });
        }
    };
    report.method = "normal_offset_cubic";
    TopologyAnalysis after;
    if (normal_ok) {
        rebuild_completed_views();
        after = analyze_topology(completed_views, lane_width, config);
    } else {
        after.trigger_pairs.push_back("normal_offset_fit_failed");
    }
    if (!after.trigger_pairs.empty()) {
        report.method = "shared_shape_vertical_offset_fallback";
        completed.clear();
        for (int lane_id = 0; lane_id < 4; ++lane_id) {
            CubicFit fit = *reference.fit;
            fit.coefficients[0] += offsets[lane_id];
            fit.rmse = 0.0;
            completed[lane_id] = fit;
        }
        rebuild_completed_views();
        after = analyze_topology(completed_views, lane_width, config);
    }
    report.validation_passed = after.trigger_pairs.empty()
        && after.all_positive;
    report.reference_lane = reference.lane_id;
    report.reference_score = reference.score;
    if (!report.validation_passed) {
        report.failure = "completed topology did not pass ordering validation";
        return;
    }

    std::vector<BevLane> output;
    output.reserve(4U);
    for (int lane_id = 0; lane_id < 4; ++lane_id) {
        const BevLane* measured = nullptr;
        for (const BevLane& lane : lanes) {
            if (lane.lane_id == lane_id && lane.fit_accepted) {
                measured = &lane;
                break;
            }
        }
        BevLane lane;
        if (measured != nullptr) {
            lane = *measured;
            lane.source_fit = measured->fit;
            lane.has_source_fit = true;
        } else {
            lane.lane_id = lane_id;
            lane.role = role_for_lane(lane_id);
            lane.score = 0.0;
            lane.synthetic = true;
            report.synthetic_lane_ids.push_back(lane_id);
        }
        lane.fit = completed.at(lane_id);
        lane.fit_accepted = true;
        lane.funnel_clipped = false;
        lane.parallel_repaired = true;
        lane.parallel_reference_lane = reference.lane_id;
        lane.parallel_offset_m = offsets.at(lane_id);
        lane.parallel_repair_method = report.method;
        output.push_back(std::move(lane));
    }
    lanes = std::move(output);
    report.applied = true;
}

bool clip_to_funnel(CubicFit& fit, double margin, bool& clipped) {
    clipped = false;
    if (!fit.valid || fit.x_max - fit.x_min < 1.0) {
        return false;
    }
    const int count = std::max(
        3, static_cast<int>(std::ceil((fit.x_max - fit.x_min) / 0.5)) + 1
    );
    std::vector<double> x(static_cast<std::size_t>(count));
    std::vector<std::uint8_t> valid(static_cast<std::size_t>(count), 0U);
    const double tangent = std::tan(15.0 * kPi / 180.0);
    for (int index = 0; index < count; ++index) {
        x[static_cast<std::size_t>(index)] = fit.x_min
            + (fit.x_max - fit.x_min) * static_cast<double>(index)
                / static_cast<double>(count - 1);
        const double y = evaluate(fit.coefficients, x[static_cast<std::size_t>(index)]);
        const double half_width = std::max(
            0.0, x[static_cast<std::size_t>(index)] - kCameraX
        ) * tangent;
        valid[static_cast<std::size_t>(index)] = static_cast<std::uint8_t>(
            std::isfinite(y)
            && x[static_cast<std::size_t>(index)] >= kCameraX
            && std::abs(y) <= half_width + margin
        );
    }
    int best_start = -1;
    int best_end = -1;
    int start = -1;
    for (int index = 0; index < count; ++index) {
        if (valid[static_cast<std::size_t>(index)] != 0U && start < 0) {
            start = index;
        }
        if (start >= 0 && (valid[static_cast<std::size_t>(index)] == 0U
                          || index == count - 1)) {
            const int end = valid[static_cast<std::size_t>(index)] != 0U
                && index == count - 1 ? index : index - 1;
            if (x[static_cast<std::size_t>(end)] - x[static_cast<std::size_t>(start)] >= 1.0
                && (best_start < 0
                    || x[static_cast<std::size_t>(end)] - x[static_cast<std::size_t>(start)]
                        > x[static_cast<std::size_t>(best_end)] - x[static_cast<std::size_t>(best_start)])) {
                best_start = start;
                best_end = end;
            }
            start = -1;
        }
    }
    if (best_start < 0) {
        return false;
    }
    const double original_min = fit.x_min;
    const double original_max = fit.x_max;
    fit.x_min = x[static_cast<std::size_t>(best_start)];
    fit.x_max = x[static_cast<std::size_t>(best_end)];
    clipped = fit.x_min > original_min + 1e-6
        || fit.x_max < original_max - 1e-6;
    return true;
}

}  // namespace

const char* bev_mode_name(BevMode mode) {
    switch (mode) {
    case BevMode::Raw: return "raw";
    case BevMode::Trigger: return "trigger";
    case BevMode::AlwaysParallel: return "always-parallel";
    case BevMode::CompleteFour: return "complete-four";
    }
    throw std::invalid_argument("unknown BEV mode enum");
}

BevMode parse_bev_mode(const std::string& value) {
    if (value == "raw" || value == "off" || value == "none") {
        return BevMode::Raw;
    }
    if (value == "trigger" || value == "triggered") {
        return BevMode::Trigger;
    }
    if (value == "always-parallel" || value == "always_parallel") {
        return BevMode::AlwaysParallel;
    }
    if (value == "complete-four" || value == "complete_four"
        || value == "complete4") {
        return BevMode::CompleteFour;
    }
    throw std::invalid_argument(
        "invalid --bev-mode '" + value
        + "' (expected raw, trigger, always-parallel, or complete-four)"
    );
}

void apply_bev_mode(
    std::vector<BevLane>& lanes,
    const BevConfig& config,
    BevTopologyReport* topology
) {
    BevTopologyReport report;
    report.mode = config.mode;
    report.lane_width_m = config.nominal_lane_width;
    for (BevLane& lane : lanes) {
        lane.funnel_clipped = false;
        lane.parallel_repaired = false;
        lane.synthetic = false;
        lane.has_source_fit = false;
        lane.parallel_reference_lane = -1;
        lane.parallel_offset_m = 0.0;
        lane.parallel_repair_method.clear();
    }

    switch (config.mode) {
    case BevMode::Raw:
        report.activation = "disabled";
        break;
    case BevMode::Trigger:
        repair_parallel_lanes(lanes, config, false, report);
        break;
    case BevMode::AlwaysParallel:
        repair_parallel_lanes(lanes, config, true, report);
        break;
    case BevMode::CompleteFour:
        complete_four_lanes(lanes, config, report);
        break;
    }

    if (config.mode != BevMode::CompleteFour) {
        for (BevLane& lane : lanes) {
            if (!lane.fit_accepted) {
                continue;
            }
            lane.fit_accepted = clip_to_funnel(
                lane.fit, config.funnel_margin, lane.funnel_clipped
            );
        }
    }
    if (topology != nullptr) {
        *topology = std::move(report);
    }
}

std::vector<BevLane> project_lanes_to_bev(
    const std::vector<Lane>& lanes,
    const BevConfig& config,
    BevTopologyReport* topology
) {
    std::vector<BevLane> result;
    result.reserve(lanes.size());
    for (const Lane& lane : lanes) {
        BevLane output;
        output.lane_id = lane.lane_id;
        output.role = lane.role;
        output.score = lane.score();
        output.points = project_lane(lane, config);
        output.fit = fit_cubic(output.points);
        output.fit_accepted = output.fit.valid
            && output.fit.rmse <= config.maximum_rmse;
        result.push_back(std::move(output));
    }
    apply_bev_mode(result, config, topology);
    return result;
}

std::string bev_topology_json(const BevTopologyReport& topology) {
    std::ostringstream stream;
    stream << std::setprecision(12)
           << "{\"mode\":\"" << bev_mode_name(topology.mode) << "\""
           << ",\"parallel_assumption\":"
           << (topology.mode == BevMode::Raw ? "false" : "true")
           << ",\"applied\":" << (topology.applied ? "true" : "false")
           << ",\"forced\":" << (topology.forced ? "true" : "false")
           << ",\"validation_passed\":"
           << (topology.validation_passed ? "true" : "false")
           << ",\"funnel_bypassed\":"
           << (topology.funnel_bypassed ? "true" : "false")
           << ",\"activation\":\"" << topology.activation << "\""
           << ",\"lane_width_m\":" << topology.lane_width_m
           << ",\"lane_width_source\":\""
           << topology.lane_width_source << "\""
           << ",\"reference_lane\":";
    if (topology.reference_lane < 0) {
        stream << "null";
    } else {
        stream << topology.reference_lane;
    }
    stream << ",\"reference_score\":";
    if (topology.reference_lane < 0) {
        stream << "null";
    } else {
        stream << topology.reference_score;
    }
    stream << ",\"reference_selection\":\""
           << topology.reference_selection << "\""
           << ",\"method\":\"" << topology.method << "\""
           << ",\"trigger_pairs\":[";
    for (std::size_t index = 0; index < topology.trigger_pairs.size(); ++index) {
        if (index != 0U) {
            stream << ',';
        }
        stream << '\"' << topology.trigger_pairs[index] << '\"';
    }
    stream << "],\"synthetic_lane_ids\":[";
    for (std::size_t index = 0;
         index < topology.synthetic_lane_ids.size(); ++index) {
        if (index != 0U) {
            stream << ',';
        }
        stream << topology.synthetic_lane_ids[index];
    }
    stream << "],\"failure\":\"" << topology.failure << "\"}";
    return stream.str();
}

void write_bev_json(
    const std::string& path,
    const std::vector<BevLane>& lanes,
    const BevTopologyReport* topology
) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot write BEV JSON: " + path);
    }
    BevTopologyReport raw_report;
    const BevTopologyReport& report = topology == nullptr
        ? raw_report : *topology;
    stream << std::setprecision(12) << "{\n  \"mode\": \""
           << bev_mode_name(report.mode) << "\",\n"
           << "  \"parallel_assumption\": "
           << (report.mode == BevMode::Raw ? "false" : "true") << ",\n"
           << "  \"synthetic_lanes\": "
           << (report.synthetic_lane_ids.empty() ? "false" : "true")
           << ",\n  \"topology\": " << bev_topology_json(report)
           << ",\n  \"lanes\": [\n";
    for (std::size_t index = 0; index < lanes.size(); ++index) {
        const BevLane& lane = lanes[index];
        stream << "    {\"lane_id\": " << lane.lane_id
               << ", \"role\": \"" << lane.role
               << "\", \"score\": " << lane.score
               << ", \"projected_point_count\": " << lane.points.size()
               << ", \"valid_fit\": " << (lane.fit_accepted ? "true" : "false")
               << ", \"synthetic\": " << (lane.synthetic ? "true" : "false")
               << ", \"parallel_repaired\": "
               << (lane.parallel_repaired ? "true" : "false");
        if (lane.parallel_repaired) {
            stream << ", \"parallel_reference_lane\": "
                   << lane.parallel_reference_lane
                   << ", \"parallel_offset_m\": " << lane.parallel_offset_m
                   << ", \"parallel_repair_method\": \""
                   << lane.parallel_repair_method << "\"";
        }
        if (lane.fit.valid) {
            stream << ", \"coefficients_c0_to_c3\": ["
                   << lane.fit.coefficients[0] << ',' << lane.fit.coefficients[1]
                   << ',' << lane.fit.coefficients[2] << ',' << lane.fit.coefficients[3]
                   << "], \"x_domain_m\": [" << lane.fit.x_min << ',' << lane.fit.x_max
                   << "], \"rmse_m\": " << lane.fit.rmse
                   << ", \"inlier_count\": " << lane.fit.inlier_count
                   << ", \"funnel_clipped\": "
                   << (lane.funnel_clipped ? "true" : "false");
        }
        if (lane.has_source_fit) {
            stream << ", \"source_coefficients_c0_to_c3\": ["
                   << lane.source_fit.coefficients[0] << ','
                   << lane.source_fit.coefficients[1] << ','
                   << lane.source_fit.coefficients[2] << ','
                   << lane.source_fit.coefficients[3] << ']';
        }
        stream << '}' << (index + 1 == lanes.size() ? "\n" : ",\n");
    }
    stream << "  ]\n}\n";
}

}  // namespace rclane
