#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <omp.h>

namespace rclane {

void normalize_bgr_to_nchw(
    const std::uint8_t* bgr,
    int source_width,
    int source_height,
    std::vector<float>& destination
) {
    constexpr int output_width = 800;
    constexpr int output_height = 320;
    constexpr float mean[3]{0.485F, 0.456F, 0.406F};
    constexpr float standard_deviation[3]{0.229F, 0.224F, 0.225F};
    if (bgr == nullptr || source_width <= 0 || source_height <= 0) {
        throw std::invalid_argument("invalid BGR source image");
    }
    const std::size_t plane = static_cast<std::size_t>(
        output_width * output_height
    );
    destination.resize(plane * 3);
    constexpr int coefficient_scale = 1 << 11;
    const double scale_x = static_cast<double>(source_width)
        / static_cast<double>(output_width);
    const double scale_y = static_cast<double>(source_height)
        / static_cast<double>(output_height);
#pragma omp parallel for schedule(static)
    for (int output_y = 0; output_y < output_height; ++output_y) {
        const float source_y = static_cast<float>(
            (static_cast<double>(output_y) + 0.5) * scale_y - 0.5
        );
        int y0 = static_cast<int>(std::floor(source_y));
        float wy = source_y - static_cast<float>(y0);
        if (y0 < 0) {
            y0 = 0;
            wy = 0.0F;
        }
        int y1 = std::min(y0 + 1, source_height - 1);
        if (y0 >= source_height - 1) {
            y0 = source_height - 1;
            y1 = y0;
            wy = 0.0F;
        }
        const int beta1 = static_cast<int>(std::nearbyint(
            wy * static_cast<float>(coefficient_scale)
        ));
        const int beta0 = static_cast<int>(std::nearbyint(
            (1.0F - wy) * static_cast<float>(coefficient_scale)
        ));
        for (int output_x = 0; output_x < output_width; ++output_x) {
            const float source_x = static_cast<float>(
                (static_cast<double>(output_x) + 0.5) * scale_x - 0.5
            );
            int x0 = static_cast<int>(std::floor(source_x));
            float wx = source_x - static_cast<float>(x0);
            if (x0 < 0) {
                x0 = 0;
                wx = 0.0F;
            }
            int x1 = std::min(x0 + 1, source_width - 1);
            if (x0 >= source_width - 1) {
                x0 = source_width - 1;
                x1 = x0;
                wx = 0.0F;
            }
            const int alpha1 = static_cast<int>(std::nearbyint(
                wx * static_cast<float>(coefficient_scale)
            ));
            const int alpha0 = static_cast<int>(std::nearbyint(
                (1.0F - wx) * static_cast<float>(coefficient_scale)
            ));
            const std::size_t output = static_cast<std::size_t>(
                output_y * output_width + output_x
            );
            const std::size_t top_left = static_cast<std::size_t>(
                (y0 * source_width + x0) * 3
            );
            const std::size_t top_right = static_cast<std::size_t>(
                (y0 * source_width + x1) * 3
            );
            const std::size_t bottom_left = static_cast<std::size_t>(
                (y1 * source_width + x0) * 3
            );
            const std::size_t bottom_right = static_cast<std::size_t>(
                (y1 * source_width + x1) * 3
            );
            for (int rgb_channel = 0; rgb_channel < 3; ++rgb_channel) {
                const int bgr_channel = 2 - rgb_channel;
                const int top = static_cast<int>(bgr[top_left + bgr_channel]) * alpha0
                    + static_cast<int>(bgr[top_right + bgr_channel]) * alpha1;
                const int bottom = static_cast<int>(bgr[bottom_left + bgr_channel]) * alpha0
                    + static_cast<int>(bgr[bottom_right + bgr_channel]) * alpha1;
                // Match OpenCV 4.11's Apache-licensed VResizeLinear<8u>; see
                // cpp/THIRD_PARTY_NOTICES.md. Its
                // fixed-point shifts intentionally happen before accumulation.
                const int pixel_integer = (
                    ((beta0 * (top >> 4)) >> 16)
                    + ((beta1 * (bottom >> 4)) >> 16) + 2
                ) >> 2;
                const float pixel = static_cast<float>(
                    std::clamp(pixel_integer, 0, 255)
                );
                destination[static_cast<std::size_t>(rgb_channel) * plane + output]
                    = (pixel * (1.0F / 255.0F) - mean[rgb_channel])
                        / standard_deviation[rgb_channel];
            }
        }
    }
}

}  // namespace rclane
