#include "decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <omp.h>

namespace rclane {
namespace {

constexpr int kMapWidth = 800;
constexpr int kMapHeight = 320;

struct Seed {
    int x{};
    int y{};
    float probability{};
};

std::vector<std::size_t> numpy_float_argquicksort(
    const std::vector<Seed>& candidates
) {
    // Partition structure follows NumPy's BSD-licensed npysort aquicksort;
    // see cpp/THIRD_PARTY_NOTICES.md.
    // Match NumPy 1.26's default np.argsort quicksort, including its
    // deterministic (but unstable) ordering of equal saturated probabilities.
    // This matters because the segmentation map contains long probability=1
    // plateaus and greedy point-NMS consumes candidates in argsort order.
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (order.size() <= 1) {
        return order;
    }
    const auto key = [&candidates](std::size_t index) {
        return -candidates[index].probability;
    };
    const auto less = [&key](std::size_t lhs, std::size_t rhs) {
        return key(lhs) < key(rhs);
    };
    struct Partition {
        std::ptrdiff_t left{};
        std::ptrdiff_t right{};
        int depth{};
    };
    std::vector<Partition> stack;
    stack.reserve(128);
    std::ptrdiff_t left = 0;
    std::ptrdiff_t right = static_cast<std::ptrdiff_t>(order.size() - 1);
    int most_significant_bit = 0;
    for (std::size_t size = order.size(); size > 1; size >>= 1U) {
        ++most_significant_bit;
    }
    int depth = most_significant_bit * 2;
    for (;;) {
        if (depth < 0) {
            // NumPy switches to arg-heapsort here. This branch is not reached
            // by the map sizes/distributions used by RCLane; retain a safe
            // deterministic fallback for adversarial inputs.
            std::sort(
                order.begin() + left, order.begin() + right + 1, less
            );
            goto pop_partition;
        }
        while (right - left > 15) {
            const std::ptrdiff_t middle = left + ((right - left) >> 1);
            if (less(order[static_cast<std::size_t>(middle)],
                     order[static_cast<std::size_t>(left)])) {
                std::swap(order[static_cast<std::size_t>(middle)],
                          order[static_cast<std::size_t>(left)]);
            }
            if (less(order[static_cast<std::size_t>(right)],
                     order[static_cast<std::size_t>(middle)])) {
                std::swap(order[static_cast<std::size_t>(right)],
                          order[static_cast<std::size_t>(middle)]);
            }
            if (less(order[static_cast<std::size_t>(middle)],
                     order[static_cast<std::size_t>(left)])) {
                std::swap(order[static_cast<std::size_t>(middle)],
                          order[static_cast<std::size_t>(left)]);
            }
            const float pivot = key(order[static_cast<std::size_t>(middle)]);
            std::ptrdiff_t i = left;
            std::ptrdiff_t j = right - 1;
            std::swap(order[static_cast<std::size_t>(middle)],
                      order[static_cast<std::size_t>(j)]);
            for (;;) {
                do {
                    ++i;
                } while (key(order[static_cast<std::size_t>(i)]) < pivot);
                do {
                    --j;
                } while (pivot < key(order[static_cast<std::size_t>(j)]));
                if (i >= j) {
                    break;
                }
                std::swap(order[static_cast<std::size_t>(i)],
                          order[static_cast<std::size_t>(j)]);
            }
            std::swap(order[static_cast<std::size_t>(i)],
                      order[static_cast<std::size_t>(right - 1)]);
            --depth;
            if (i - left < right - i) {
                stack.push_back({i + 1, right, depth});
                right = i - 1;
            } else {
                stack.push_back({left, i - 1, depth});
                left = i + 1;
            }
        }
        for (std::ptrdiff_t i = left + 1; i <= right; ++i) {
            const std::size_t value = order[static_cast<std::size_t>(i)];
            std::ptrdiff_t position = i;
            std::ptrdiff_t previous = i - 1;
            while (position > left
                   && key(value) < key(order[static_cast<std::size_t>(previous)])) {
                order[static_cast<std::size_t>(position--)]
                    = order[static_cast<std::size_t>(previous--)];
            }
            order[static_cast<std::size_t>(position)] = value;
        }
    pop_partition:
        if (stack.empty()) {
            break;
        }
        const Partition next = stack.back();
        stack.pop_back();
        left = next.left;
        right = next.right;
        depth = next.depth;
    }
    return order;
}

float median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0F;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const float upper = values[middle];
    if ((values.size() & 1U) != 0U) {
        return upper;
    }
    const float lower = *std::max_element(
        values.begin(), values.begin() + middle
    );
    return (lower + upper) * 0.5F;
}

double quantile(std::vector<float> values, double fraction) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double blend = position - static_cast<double>(lower);
    return static_cast<double>(values[lower]) * (1.0 - blend)
        + static_cast<double>(values[upper]) * blend;
}

std::vector<Seed> select_seeds(
    const std::vector<float>& probability,
    const DecoderConfig& config,
    DecodeStatistics* statistics
) {
    std::vector<Seed> candidates;
    candidates.reserve(probability.size() / 20);
    for (int y = 0; y < kMapHeight; ++y) {
        for (int x = 0; x < kMapWidth; ++x) {
            const float value = probability[static_cast<std::size_t>(
                y * kMapWidth + x
            )];
            if (value > config.seed_threshold) {
                candidates.push_back({x, y, value});
            }
        }
    }
    if (statistics != nullptr) {
        statistics->foreground_pixels = candidates.size();
    }
    const auto order = numpy_float_argquicksort(candidates);

    std::vector<std::uint8_t> taken(
        static_cast<std::size_t>(kMapWidth * kMapHeight), 0
    );
    std::vector<Seed> selected;
    selected.reserve(static_cast<std::size_t>(config.max_seeds));
    for (const std::size_t candidate_index : order) {
        const Seed& candidate = candidates[candidate_index];
        const std::size_t position = static_cast<std::size_t>(
            candidate.y * kMapWidth + candidate.x
        );
        if (taken[position] != 0U) {
            continue;
        }
        selected.push_back(candidate);
        const int y0 = std::max(0, candidate.y - config.seed_min_distance);
        const int y1 = std::min(
            kMapHeight - 1, candidate.y + config.seed_min_distance
        );
        const int x0 = std::max(0, candidate.x - config.seed_min_distance);
        const int x1 = std::min(
            kMapWidth - 1, candidate.x + config.seed_min_distance
        );
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                taken[static_cast<std::size_t>(y * kMapWidth + x)] = 1U;
            }
        }
        if (static_cast<int>(selected.size()) >= config.max_seeds) {
            break;
        }
    }
    if (statistics != nullptr) {
        statistics->seeds = selected.size();
    }
    return selected;
}

std::vector<LanePoint> crawl(
    const Seed& seed,
    const std::vector<float>& probability,
    const std::vector<float>& arrow,
    const std::vector<float>& bound,
    const DecoderConfig& config
) {
    std::vector<LanePoint> points;
    points.reserve(48);
    int cx = seed.x;
    int cy = seed.y;
    double remain_square_sum = 0.0;
    int remain_count = 0;
    const std::size_t channel_elements = static_cast<std::size_t>(
        kMapWidth * kMapHeight
    );
    for (int index = 0; index < kMapHeight; ++index) {
        const std::size_t current = static_cast<std::size_t>(
            cy * kMapWidth + cx
        );
        if (probability[current] > config.segmentation_threshold) {
            const double remain = static_cast<double>(bound[current]) * 100.0
                / static_cast<double>(config.step_length)
                + static_cast<double>(index);
            remain_square_sum += remain * remain;
            ++remain_count;
        }
        const float dx = arrow[current];
        const float dy = arrow[channel_elements + current];
        const float norm = std::sqrt(dx * dx + dy * dy);
        if (norm == 0.0F || !std::isfinite(norm)) {
            break;
        }
        cx = static_cast<int>(std::floor(
            static_cast<float>(cx) + dx / norm * config.step_length
        ));
        cy = static_cast<int>(std::floor(
            static_cast<float>(cy) + dy / norm * config.step_length
        ));
        if (cx < 0 || cx >= kMapWidth || cy < 0 || cy >= kMapHeight) {
            break;
        }
        const float score = probability[static_cast<std::size_t>(
            cy * kMapWidth + cx
        )];
        points.push_back({static_cast<float>(cx), static_cast<float>(cy), score});
        const double remaining = remain_count > 0
            ? std::sqrt(remain_square_sum / static_cast<double>(remain_count))
            : 1.0;
        if (score > config.segmentation_threshold) {
            continue;
        }
        if (static_cast<double>(index) > remaining * 0.75) {
            break;
        }
    }
    return points;
}

double reference_x(const Lane& lane) {
    if (lane.points.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    if (lane.points.size() < 2) {
        return lane.points.front().x;
    }
    std::vector<float> y_values;
    y_values.reserve(lane.points.size());
    for (const auto& point : lane.points) {
        if (std::isfinite(point.x) && std::isfinite(point.y)) {
            y_values.push_back(point.y);
        }
    }
    if (y_values.size() < 2) {
        return lane.points.front().x;
    }
    const double cutoff = quantile(y_values, 0.6);
    double sum_x = 0.0;
    double sum_y = 0.0;
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    std::vector<const LanePoint*> lower;
    for (const auto& point : lane.points) {
        if (std::isfinite(point.x) && std::isfinite(point.y)
            && static_cast<double>(point.y) >= cutoff) {
            lower.push_back(&point);
            sum_x += point.x;
            sum_y += point.y;
            min_y = std::min(min_y, static_cast<double>(point.y));
            max_y = std::max(max_y, static_cast<double>(point.y));
        }
    }
    const auto bottom_x = [&lane]() {
        return static_cast<double>(std::max_element(
            lane.points.begin(), lane.points.end(),
            [](const LanePoint& lhs, const LanePoint& rhs) {
                return lhs.y < rhs.y;
            }
        )->x);
    };
    if (lower.size() < 2 || max_y - min_y < 1.0) {
        return bottom_x();
    }
    const double mean_x = sum_x / static_cast<double>(lower.size());
    const double mean_y = sum_y / static_cast<double>(lower.size());
    double numerator = 0.0;
    double denominator = 0.0;
    for (const auto* point : lower) {
        const double centered_y = static_cast<double>(point->y) - mean_y;
        numerator += centered_y * (static_cast<double>(point->x) - mean_x);
        denominator += centered_y * centered_y;
    }
    if (denominator <= 1e-6) {
        return bottom_x();
    }
    return mean_x + numerator / denominator
        * (static_cast<double>(lane.height - 1) - mean_y);
}

std::vector<int> preselect_candidates(
    const std::vector<Lane>& lanes,
    const std::vector<int>& score_order,
    int max_lanes
) {
    if (static_cast<int>(score_order.size()) <= max_lanes) {
        return score_order;
    }
    struct Bucket {
        int key{};
        std::vector<int> indices;
    };
    std::vector<Bucket> buckets;
    for (const int index : score_order) {
        const Lane& lane = lanes[static_cast<std::size_t>(index)];
        std::vector<float> y;
        y.reserve(lane.points.size());
        for (const auto& point : lane.points) {
            y.push_back(point.y);
        }
        const float median_y = median(std::move(y));
        std::vector<float> lower_x;
        for (const auto& point : lane.points) {
            if (point.y >= median_y) {
                lower_x.push_back(point.x);
            }
        }
        const int key = static_cast<int>(std::floor(median(lower_x) / 16.0F));
        auto found = std::find_if(
            buckets.begin(), buckets.end(),
            [key](const Bucket& bucket) { return bucket.key == key; }
        );
        if (found == buckets.end()) {
            buckets.push_back({key, {index}});
        } else {
            found->indices.push_back(index);
        }
    }
    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(max_lanes));
    for (std::size_t rank = 0; static_cast<int>(selected.size()) < max_lanes;
         ++rank) {
        bool progressed = false;
        for (const auto& bucket : buckets) {
            if (rank < bucket.indices.size()) {
                selected.push_back(bucket.indices[rank]);
                progressed = true;
                if (static_cast<int>(selected.size()) == max_lanes) {
                    break;
                }
            }
        }
        if (!progressed) {
            break;
        }
    }
    std::stable_sort(selected.begin(), selected.end(), [&lanes](int lhs, int rhs) {
        return lanes[static_cast<std::size_t>(lhs)].score()
            > lanes[static_cast<std::size_t>(rhs)].score();
    });
    return selected;
}

void draw_disk(
    std::vector<std::uint8_t>& mask, int width, int height,
    int cx, int cy, int radius
) {
    for (int y = std::max(0, cy - radius); y <= std::min(height - 1, cy + radius); ++y) {
        for (int x = std::max(0, cx - radius); x <= std::min(width - 1, cx + radius); ++x) {
            const int dx = x - cx;
            const int dy = y - cy;
            if (dx * dx + dy * dy <= radius * radius) {
                mask[static_cast<std::size_t>(y * width + x)] = 1U;
            }
        }
    }
}

std::vector<std::uint64_t> rasterize(
    const Lane& lane, float scale, int lane_width, int width, int height
) {
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(width * height), 0U
    );
    if (lane.points.size() < 2) {
        return std::vector<std::uint64_t>(
            (pixels.size() + 63U) / 64U, 0U
        );
    }
    const int thickness = std::max(1, static_cast<int>(std::lround(
        static_cast<double>(lane_width) * scale
    )));
    const int radius = std::max(1, thickness / 2);
    for (std::size_t index = 1; index < lane.points.size(); ++index) {
        int x0 = std::clamp(static_cast<int>(lane.points[index - 1].x * scale), 0, width - 1);
        int y0 = std::clamp(static_cast<int>(lane.points[index - 1].y * scale), 0, height - 1);
        const int x1 = std::clamp(static_cast<int>(lane.points[index].x * scale), 0, width - 1);
        const int y1 = std::clamp(static_cast<int>(lane.points[index].y * scale), 0, height - 1);
        const int dx = std::abs(x1 - x0);
        const int sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0);
        const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        for (;;) {
            draw_disk(pixels, width, height, x0, y0, radius);
            if (x0 == x1 && y0 == y1) {
                break;
            }
            const int twice = 2 * error;
            if (twice >= dy) {
                error += dy;
                x0 += sx;
            }
            if (twice <= dx) {
                error += dx;
                y0 += sy;
            }
        }
    }
    std::vector<std::uint64_t> mask((pixels.size() + 63U) / 64U, 0U);
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        if (pixels[index] != 0U) {
            mask[index / 64U] |= std::uint64_t{1} << (index % 64U);
        }
    }
    return mask;
}

std::vector<Lane> nms(
    std::vector<Lane> lanes,
    const DecoderConfig& config,
    DecodeStatistics* statistics
) {
    std::vector<int> order(lanes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&lanes](int lhs, int rhs) {
        return lanes[static_cast<std::size_t>(lhs)].score()
            > lanes[static_cast<std::size_t>(rhs)].score();
    });
    order = preselect_candidates(lanes, order, config.nms_max_lanes);
    if (statistics != nullptr) {
        statistics->nms_candidates = order.size();
    }
    const int width = std::max(1, static_cast<int>(std::lround(
        static_cast<double>(kMapWidth) * config.nms_scale
    )));
    const int height = std::max(1, static_cast<int>(std::lround(
        static_cast<double>(kMapHeight) * config.nms_scale
    )));
    std::vector<std::vector<std::uint64_t>> masks;
    std::vector<int> areas;
    masks.reserve(order.size());
    areas.reserve(order.size());
    for (const int index : order) {
        masks.push_back(rasterize(
            lanes[static_cast<std::size_t>(index)], config.nms_scale,
            config.lane_width, width, height
        ));
        int area = 0;
        for (const std::uint64_t word : masks.back()) {
            area += __builtin_popcountll(word);
        }
        areas.push_back(area);
    }
    std::vector<std::uint8_t> suppressed(order.size(), 0U);
    std::vector<Lane> kept;
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (suppressed[i] != 0U) {
            continue;
        }
        kept.push_back(std::move(lanes[static_cast<std::size_t>(order[i])])) ;
        for (std::size_t j = i + 1; j < order.size(); ++j) {
            if (suppressed[j] != 0U) {
                continue;
            }
            int intersection = 0;
            for (std::size_t word = 0; word < masks[i].size(); ++word) {
                intersection += __builtin_popcountll(
                    masks[i][word] & masks[j][word]
                );
            }
            const int union_area = areas[i] + areas[j] - intersection;
            if (union_area > 0
                && static_cast<double>(intersection) / union_area
                    >= config.iou_threshold) {
                suppressed[j] = 1U;
            }
        }
    }
    if (statistics != nullptr) {
        statistics->nms_survivors = kept.size();
    }
    return kept;
}

void assign_roles(std::vector<Lane>& lanes, double ego_x) {
    std::vector<Lane*> left;
    std::vector<Lane*> right;
    std::unordered_map<const Lane*, double> references;
    for (auto& lane : lanes) {
        lane.lane_id = std::numeric_limits<int>::min();
        lane.role.clear();
        lane.ego_boundary = false;
        lane.lateral_rank = 0;
        references[&lane] = reference_x(lane);
        (references[&lane] < ego_x ? left : right).push_back(&lane);
    }
    const auto near_ego = [&references, ego_x](const Lane* lhs, const Lane* rhs) {
        const double dl = std::abs(references[lhs] - ego_x);
        const double dr = std::abs(references[rhs] - ego_x);
        return dl != dr ? dl < dr : lhs->score() > rhs->score();
    };
    std::sort(left.begin(), left.end(), near_ego);
    std::sort(right.begin(), right.end(), near_ego);
    for (std::size_t index = 0; index < left.size(); ++index) {
        const int rank = static_cast<int>(index + 1);
        left[index]->lane_id = 2 - rank;
        left[index]->lateral_rank = -rank;
        left[index]->ego_boundary = rank == 1;
        left[index]->role = rank == 1 ? "ego_left" : "left_" + std::to_string(rank);
    }
    for (std::size_t index = 0; index < right.size(); ++index) {
        const int rank = static_cast<int>(index + 1);
        right[index]->lane_id = 1 + rank;
        right[index]->lateral_rank = rank;
        right[index]->ego_boundary = rank == 1;
        right[index]->role = rank == 1 ? "ego_right" : "right_" + std::to_string(rank);
    }
    std::sort(lanes.begin(), lanes.end(), [](const Lane& lhs, const Lane& rhs) {
        return lhs.lane_id < rhs.lane_id;
    });
}

std::vector<Lane> select_ego_lanes(
    std::vector<Lane> lanes, const DecoderConfig& config
) {
    if (lanes.empty()) {
        return {};
    }
    std::vector<int> pool(lanes.size());
    std::iota(pool.begin(), pool.end(), 0);
    std::vector<double> references(lanes.size());
    for (std::size_t index = 0; index < lanes.size(); ++index) {
        references[index] = reference_x(lanes[index]);
    }
    if (static_cast<int>(lanes.size()) > config.max_output_lanes) {
        const double best = std::max_element(
            lanes.begin(), lanes.end(), [](const Lane& lhs, const Lane& rhs) {
                return lhs.score() < rhs.score();
            }
        )->score();
        std::vector<int> reliable;
        for (const int index : pool) {
            if (lanes[static_cast<std::size_t>(index)].score()
                >= best * config.ego_min_score_ratio) {
                reliable.push_back(index);
            }
        }
        if (static_cast<int>(reliable.size()) >= config.max_output_lanes) {
            pool = std::move(reliable);
        }
    }
    const auto proximity = [&lanes, &references, &config](int lhs, int rhs) {
        const double dl = std::abs(references[static_cast<std::size_t>(lhs)] - config.ego_x);
        const double dr = std::abs(references[static_cast<std::size_t>(rhs)] - config.ego_x);
        return dl != dr ? dl < dr
            : lanes[static_cast<std::size_t>(lhs)].score()
                > lanes[static_cast<std::size_t>(rhs)].score();
    };
    std::vector<int> left;
    std::vector<int> right;
    for (const int index : pool) {
        (references[static_cast<std::size_t>(index)] < config.ego_x
            ? left : right).push_back(index);
    }
    std::sort(left.begin(), left.end(), proximity);
    std::sort(right.begin(), right.end(), proximity);
    std::vector<int> selected;
    if (config.max_output_lanes == 4) {
        selected.insert(selected.end(), left.begin(), left.begin() + std::min<std::size_t>(2, left.size()));
        selected.insert(selected.end(), right.begin(), right.begin() + std::min<std::size_t>(2, right.size()));
    } else {
        std::sort(pool.begin(), pool.end(), proximity);
        pool.resize(std::min<std::size_t>(pool.size(), static_cast<std::size_t>(config.max_output_lanes)));
        selected = std::move(pool);
    }
    std::vector<Lane> result;
    result.reserve(selected.size());
    for (const int index : selected) {
        result.push_back(std::move(lanes[static_cast<std::size_t>(index)]));
    }
    assign_roles(result, config.ego_x);
    return result;
}

const Tensor& require_output(
    const std::unordered_map<std::string, Tensor>& outputs,
    const std::string& name
) {
    const auto found = outputs.find(name);
    if (found == outputs.end()) {
        throw std::runtime_error("missing TensorRT output: " + name);
    }
    return found->second;
}

}  // namespace

double Lane::score() const {
    return points.empty() ? 0.0
        : score_sum / static_cast<double>(points.size());
}

std::vector<float> softmax_foreground(const Tensor& logits) {
    const std::size_t plane = static_cast<std::size_t>(kMapWidth * kMapHeight);
    if (logits.values.size() != plane * 2) {
        throw std::invalid_argument("seg_map must have shape (1,2,320,800)");
    }
    std::vector<float> probability(plane);
#pragma omp parallel for schedule(static)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(plane); ++index) {
        const float difference = logits.values[static_cast<std::size_t>(index)]
            - logits.values[plane + static_cast<std::size_t>(index)];
        probability[static_cast<std::size_t>(index)]
            = 1.0F / (1.0F + std::exp(difference));
    }
    return probability;
}

std::vector<Lane> decode(
    const std::vector<float>& probability,
    const Tensor& up_arrow,
    const Tensor& down_arrow,
    const Tensor& up_bound,
    const Tensor& down_bound,
    const DecoderConfig& config,
    DecodeStatistics* statistics
) {
    const std::size_t plane = static_cast<std::size_t>(kMapWidth * kMapHeight);
    if (probability.size() != plane
        || up_arrow.values.size() != plane * 2
        || down_arrow.values.size() != plane * 2
        || up_bound.values.size() != plane * 2
        || down_bound.values.size() != plane * 2) {
        throw std::invalid_argument("decoder map shape mismatch");
    }
    if (statistics != nullptr) {
        *statistics = {};
    }
    omp_set_num_threads(config.threads);
    const auto seeds = select_seeds(probability, config, statistics);
    std::vector<std::vector<LanePoint>> up(seeds.size());
    std::vector<std::vector<LanePoint>> down(seeds.size());
#pragma omp parallel for schedule(static)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(seeds.size()); ++index) {
        up[static_cast<std::size_t>(index)] = crawl(
            seeds[static_cast<std::size_t>(index)], probability,
            up_arrow.values, up_bound.values, config
        );
        down[static_cast<std::size_t>(index)] = crawl(
            seeds[static_cast<std::size_t>(index)], probability,
            down_arrow.values, down_bound.values, config
        );
    }
    std::vector<Lane> candidates;
    candidates.reserve(seeds.size());
    for (std::size_t index = 0; index < seeds.size(); ++index) {
        const std::size_t count = up[index].size() + down[index].size();
        if (count <= 1) {
            continue;
        }
        Lane lane;
        lane.width = kMapWidth;
        lane.height = kMapHeight;
        lane.points.reserve(count);
        for (auto point = up[index].rbegin(); point != up[index].rend(); ++point) {
            lane.points.push_back(*point);
            lane.score_sum += point->score;
        }
        for (const auto& point : down[index]) {
            lane.points.push_back(point);
            lane.score_sum += point.score;
        }
        if (lane.score() >= config.score_threshold) {
            candidates.push_back(std::move(lane));
        }
    }
    if (statistics != nullptr) {
        statistics->crawled_candidates = candidates.size();
    }
    auto kept = nms(std::move(candidates), config, statistics);
    return select_ego_lanes(std::move(kept), config);
}

std::vector<Lane> decode_outputs(
    const std::unordered_map<std::string, Tensor>& outputs,
    const DecoderConfig& config,
    DecodeStatistics* statistics
) {
    const auto probability = softmax_foreground(require_output(outputs, "seg_map"));
    return decode(
        probability,
        require_output(outputs, "up_arrow"),
        require_output(outputs, "down_arrow"),
        require_output(outputs, "up_bound"),
        require_output(outputs, "down_bound"),
        config,
        statistics
    );
}

void write_lanes_json(
    const std::string& path,
    const std::vector<Lane>& lanes,
    const DecodeStatistics* statistics
) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot write lanes JSON: " + path);
    }
    stream << std::setprecision(9) << "{\n";
    if (statistics != nullptr) {
        stream << "  \"statistics\": {\"foreground_pixels\": "
               << statistics->foreground_pixels << ", \"seeds\": "
               << statistics->seeds << ", \"crawled_candidates\": "
               << statistics->crawled_candidates << ", \"nms_candidates\": "
               << statistics->nms_candidates << ", \"nms_survivors\": "
               << statistics->nms_survivors << "},\n";
    }
    stream << "  \"lanes\": [\n";
    for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
        const Lane& lane = lanes[lane_index];
        stream << "    {\"lane_id\": " << lane.lane_id
               << ", \"role\": \"" << lane.role
               << "\", \"score\": " << lane.score() << ", \"points\": [";
        for (std::size_t point_index = 0; point_index < lane.points.size(); ++point_index) {
            const auto& point = lane.points[point_index];
            if (point_index != 0) {
                stream << ',';
            }
            stream << '[' << point.x << ',' << point.y << ',' << point.score << ']';
        }
        stream << "]}" << (lane_index + 1 == lanes.size() ? "\n" : ",\n");
    }
    stream << "  ]\n}\n";
}

}  // namespace rclane
