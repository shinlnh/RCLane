#pragma once

#include <cstdint>
#include <vector>

namespace rclane {

void normalize_bgr_to_nchw(
    const std::uint8_t* bgr,
    int source_width,
    int source_height,
    std::vector<float>& destination
);

}  // namespace rclane
