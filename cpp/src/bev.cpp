#include "bev.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
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

std::vector<BevLane> project_lanes_to_bev(
    const std::vector<Lane>& lanes,
    const BevConfig& config
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
        if (output.fit_accepted) {
            output.fit_accepted = clip_to_funnel(
                output.fit, config.funnel_margin, output.funnel_clipped
            );
        }
        result.push_back(std::move(output));
    }
    return result;
}

void write_bev_json(
    const std::string& path,
    const std::vector<BevLane>& lanes
) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot write BEV JSON: " + path);
    }
    stream << std::setprecision(12) << "{\n  \"mode\": \"raw_model_projection\",\n"
           << "  \"parallel_assumption\": false,\n"
           << "  \"synthetic_lanes\": false,\n  \"lanes\": [\n";
    for (std::size_t index = 0; index < lanes.size(); ++index) {
        const BevLane& lane = lanes[index];
        stream << "    {\"lane_id\": " << lane.lane_id
               << ", \"role\": \"" << lane.role
               << "\", \"score\": " << lane.score
               << ", \"projected_point_count\": " << lane.points.size()
               << ", \"valid_fit\": " << (lane.fit_accepted ? "true" : "false");
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
        stream << '}' << (index + 1 == lanes.size() ? "\n" : ",\n");
    }
    stream << "  ]\n}\n";
}

}  // namespace rclane
