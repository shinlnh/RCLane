#include "tensorrt_runner.hpp"
#include "decoder.hpp"
#include "bev.hpp"
#include "preprocess.hpp"

#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Arguments {
    std::string engine;
    std::string input;
    std::string input_bgr;
    std::string dump_prefix;
    std::string lanes_json;
    std::string bev_json;
    std::string report;
    std::string frames_jsonl;
    bool raw_bgr_stdin{false};
    int source_width{1920};
    int source_height{1080};
    int max_frames{0};
    int timing_warmup{5};
    int warmup{10};
    int iterations{100};
    int threads{8};
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        const auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument("missing value for " + option);
            }
            return argv[index];
        };
        if (option == "--engine") {
            args.engine = value();
        } else if (option == "--input-nchw") {
            args.input = value();
        } else if (option == "--input-bgr") {
            args.input_bgr = value();
        } else if (option == "--raw-bgr-stdin") {
            args.raw_bgr_stdin = true;
        } else if (option == "--dump-prefix") {
            args.dump_prefix = value();
        } else if (option == "--lanes-json") {
            args.lanes_json = value();
        } else if (option == "--bev-json") {
            args.bev_json = value();
        } else if (option == "--report") {
            args.report = value();
        } else if (option == "--frames-jsonl") {
            args.frames_jsonl = value();
        } else if (option == "--source-width") {
            args.source_width = std::stoi(value());
        } else if (option == "--source-height") {
            args.source_height = std::stoi(value());
        } else if (option == "--max-frames") {
            args.max_frames = std::stoi(value());
        } else if (option == "--timing-warmup") {
            args.timing_warmup = std::stoi(value());
        } else if (option == "--warmup") {
            args.warmup = std::stoi(value());
        } else if (option == "--iterations") {
            args.iterations = std::stoi(value());
        } else if (option == "--threads") {
            args.threads = std::stoi(value());
        } else {
            throw std::invalid_argument("unknown argument: " + option);
        }
    }
    const int input_modes = static_cast<int>(!args.input.empty())
        + static_cast<int>(!args.input_bgr.empty())
        + static_cast<int>(args.raw_bgr_stdin);
    if (args.engine.empty() || input_modes != 1
        || args.source_width <= 0 || args.source_height <= 0
        || args.timing_warmup < 0 || args.max_frames < 0) {
        throw std::invalid_argument(
            "usage: rclane_runtime --engine model.engine "
            "(--input-nchw frame.f32 | --input-bgr frame.bgr | "
            "--raw-bgr-stdin) "
            "[--dump-prefix output]"
        );
    }
    return args;
}

std::vector<float> read_floats(
    const std::string& path, std::size_t expected
) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open input: " + path);
    }
    const auto byte_count = static_cast<std::size_t>(stream.tellg());
    if (byte_count != expected * sizeof(float)) {
        throw std::runtime_error(
            "input byte count mismatch: expected "
            + std::to_string(expected * sizeof(float)) + ", got "
            + std::to_string(byte_count)
        );
    }
    std::vector<float> values(expected);
    stream.seekg(0);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(byte_count)
    );
    return values;
}

std::vector<std::uint8_t> read_bytes(
    const std::string& path, std::size_t expected
) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open input: " + path);
    }
    const auto byte_count = static_cast<std::size_t>(stream.tellg());
    if (byte_count != expected) {
        throw std::runtime_error(
            "BGR byte count mismatch: expected " + std::to_string(expected)
            + ", got " + std::to_string(byte_count)
        );
    }
    std::vector<std::uint8_t> bytes(expected);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(byte_count));
    return bytes;
}

void dump_outputs(
    const std::string& prefix,
    const std::unordered_map<std::string, rclane::Tensor>& outputs
) {
    if (prefix.empty()) {
        return;
    }
    for (const auto& [name, tensor] : outputs) {
        const std::string path = prefix + "." + name + ".f32";
        std::ofstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("cannot write output: " + path);
        }
        stream.write(
            reinterpret_cast<const char*>(tensor.values.data()),
            static_cast<std::streamsize>(
                tensor.values.size() * sizeof(float)
            )
        );
    }
}

void dump_input(const std::string& prefix, const std::vector<float>& input) {
    if (prefix.empty()) {
        return;
    }
    std::ofstream stream(prefix + ".input.f32", std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot write normalized input");
    }
    stream.write(
        reinterpret_cast<const char*>(input.data()),
        static_cast<std::streamsize>(input.size() * sizeof(float))
    );
}

struct TimingSummary {
    double mean{};
    double median{};
    double p95{};
    double minimum{};
    double maximum{};
};

TimingSummary summarize(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("no timed frames after warmup");
    }
    std::sort(values.begin(), values.end());
    const auto percentile = [&values](double fraction) {
        const auto position = static_cast<std::size_t>(std::floor(
            fraction * static_cast<double>(values.size() - 1)
        ));
        return values[position];
    };
    return {
        std::accumulate(values.begin(), values.end(), 0.0)
            / static_cast<double>(values.size()),
        percentile(0.5), percentile(0.95), values.front(), values.back(),
    };
}

double milliseconds(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point stop
) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void write_stream_report(
    const std::string& path,
    std::size_t frames,
    std::size_t timed_frames,
    int threads,
    const TimingSummary& read,
    const TimingSummary& preprocess,
    const TimingSummary& inference,
    const TimingSummary& decode,
    const TimingSummary& bev,
    const TimingSummary& core,
    double mean_lanes
) {
    if (path.empty()) {
        return;
    }
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot write benchmark report: " + path);
    }
    const auto timing = [&stream](const char* name, const TimingSummary& value,
                                  bool comma) {
        stream << "    \"" << name << "\": {\"mean_ms\": " << value.mean
               << ", \"median_ms\": " << value.median
               << ", \"p95_ms\": " << value.p95
               << ", \"min_ms\": " << value.minimum
               << ", \"max_ms\": " << value.maximum << '}'
               << (comma ? ",\n" : "\n");
    };
    stream << std::setprecision(12)
           << "{\n  \"runtime\": \"native_tensorrt_cpp\",\n"
           << "  \"sequential_per_frame\": true,\n"
           << "  \"frame_overlap\": false,\n"
           << "  \"rendering_included\": false,\n"
           << "  \"video_writing_included\": false,\n"
           << "  \"bev_mode\": \"raw_model_projection\",\n"
           << "  \"parallel_assumption\": false,\n"
           << "  \"synthetic_lanes\": false,\n"
           << "  \"frames\": " << frames << ",\n"
           << "  \"timed_frames\": " << timed_frames << ",\n"
           << "  \"decode_threads\": " << threads << ",\n"
           << "  \"mean_output_lanes\": " << mean_lanes << ",\n"
           << "  \"timing\": {\n";
    timing("source_read", read, true);
    timing("preprocess", preprocess, true);
    timing("inference_with_transfers", inference, true);
    timing("decode", decode, true);
    timing("bev_cubic_funnel", bev, true);
    timing("core_pipeline", core, false);
    stream << "  },\n  \"fps_from_core_median_latency\": "
           << 1000.0 / core.median << "\n}\n";
}

int run_raw_stream(
    const Arguments& args,
    rclane::TensorRTRunner& runner
) {
    const std::size_t frame_bytes = static_cast<std::size_t>(
        args.source_width * args.source_height * 3
    );
    std::vector<std::uint8_t> frame(frame_bytes);
    std::vector<float> input(runner.input_elements(), 0.0F);
    // Warm only TensorRT with a neutral tensor; actual frame timings discard
    // the first timing_warmup frames so OpenMP and allocator startup are absent.
    for (int index = 0; index < args.warmup; ++index) {
        runner.infer_reuse(input.data());
    }
    rclane::DecoderConfig decoder_config;
    decoder_config.threads = args.threads;
    std::vector<double> read_samples;
    std::vector<double> preprocess_samples;
    std::vector<double> inference_samples;
    std::vector<double> decode_samples;
    std::vector<double> bev_samples;
    std::vector<double> core_samples;
    std::size_t frames = 0;
    std::size_t timed_frames = 0;
    double lane_sum = 0.0;
    std::ofstream frame_results;
    if (!args.frames_jsonl.empty()) {
        frame_results.open(args.frames_jsonl);
        if (!frame_results) {
            throw std::runtime_error(
                "cannot write frame results: " + args.frames_jsonl
            );
        }
        frame_results << std::setprecision(9);
    }
    for (;;) {
        if (args.max_frames > 0 && static_cast<int>(frames) >= args.max_frames) {
            break;
        }
        const auto read_start = std::chrono::steady_clock::now();
        std::cin.read(
            reinterpret_cast<char*>(frame.data()),
            static_cast<std::streamsize>(frame.size())
        );
        const auto read_stop = std::chrono::steady_clock::now();
        if (std::cin.gcount() == 0) {
            break;
        }
        if (std::cin.gcount() != static_cast<std::streamsize>(frame.size())) {
            throw std::runtime_error("partial BGR frame on stdin");
        }
        const auto core_start = std::chrono::steady_clock::now();
        const auto preprocess_start = core_start;
        rclane::normalize_bgr_to_nchw(
            frame.data(), args.source_width, args.source_height, input
        );
        const auto preprocess_stop = std::chrono::steady_clock::now();
        const auto inference_start = preprocess_stop;
        const auto& outputs = runner.infer_reuse(input.data());
        const auto inference_stop = std::chrono::steady_clock::now();
        const auto decode_start = inference_stop;
        rclane::DecodeStatistics decode_statistics;
        const auto lanes = rclane::decode_outputs(
            outputs, decoder_config, &decode_statistics
        );
        const auto decode_stop = std::chrono::steady_clock::now();
        const auto bev_lanes = rclane::project_lanes_to_bev(lanes);
        const auto bev_stop = std::chrono::steady_clock::now();
        const double frame_preprocess_ms = milliseconds(
            preprocess_start, preprocess_stop
        );
        const double frame_inference_ms = milliseconds(
            inference_start, inference_stop
        );
        const double frame_decode_ms = milliseconds(decode_start, decode_stop);
        const double frame_bev_ms = milliseconds(decode_stop, bev_stop);
        const double frame_core_ms = milliseconds(core_start, bev_stop);
        // Serialization is deliberately after bev_stop, outside core latency.
        if (frame_results) {
            frame_results << "{\"frame_index\":" << frames
                          << ",\"timing\":{\"preprocess_ms\":"
                          << frame_preprocess_ms
                          << ",\"inference_ms\":" << frame_inference_ms
                          << ",\"decode_ms\":" << frame_decode_ms
                          << ",\"bev_result_ms\":" << frame_bev_ms
                          << ",\"core_ms\":" << frame_core_ms << "}"
                          << ",\"lanes\":[";
            for (std::size_t lane_index = 0; lane_index < lanes.size();
                 ++lane_index) {
                const auto& lane = lanes[lane_index];
                if (lane_index != 0U) {
                    frame_results << ',';
                }
                frame_results << "{\"lane_id\":" << lane.lane_id
                              << ",\"role\":\"" << lane.role
                              << "\",\"score\":" << lane.score()
                              << ",\"points\":[";
                for (std::size_t point_index = 0;
                     point_index < lane.points.size(); ++point_index) {
                    const auto& point = lane.points[point_index];
                    if (point_index != 0U) {
                        frame_results << ',';
                    }
                    frame_results << '[' << point.x << ',' << point.y << ','
                                  << point.score << ']';
                }
                frame_results << "]}";
            }
            frame_results << "],\"bev_lanes\":[";
            for (std::size_t lane_index = 0;
                 lane_index < bev_lanes.size(); ++lane_index) {
                const auto& lane = bev_lanes[lane_index];
                if (lane_index != 0U) {
                    frame_results << ',';
                }
                frame_results << "{\"lane_id\":" << lane.lane_id
                              << ",\"role\":\"" << lane.role
                              << "\",\"score\":" << lane.score
                              << ",\"fit_accepted\":"
                              << (lane.fit_accepted ? "true" : "false")
                              << ",\"funnel_clipped\":"
                              << (lane.funnel_clipped ? "true" : "false")
                              << ",\"points\":[";
                for (std::size_t point_index = 0;
                     point_index < lane.points.size(); ++point_index) {
                    const auto& point = lane.points[point_index];
                    if (point_index != 0U) {
                        frame_results << ',';
                    }
                    frame_results << '[' << point.x << ',' << point.y << ','
                                  << point.score << ']';
                }
                frame_results << ']';
                if (lane.fit.valid) {
                    frame_results << ",\"fit\":{\"coefficients\":["
                                  << lane.fit.coefficients[0] << ','
                                  << lane.fit.coefficients[1] << ','
                                  << lane.fit.coefficients[2] << ','
                                  << lane.fit.coefficients[3]
                                  << "],\"x_min\":" << lane.fit.x_min
                                  << ",\"x_max\":" << lane.fit.x_max
                                  << ",\"rmse\":" << lane.fit.rmse
                                  << ",\"point_count\":"
                                  << lane.fit.point_count
                                  << ",\"inlier_count\":"
                                  << lane.fit.inlier_count << '}';
                } else {
                    frame_results << ",\"fit\":null";
                }
                frame_results << '}';
            }
            frame_results << "]}\n";
        }
        lane_sum += static_cast<double>(lanes.size());
        if (static_cast<int>(frames) >= args.timing_warmup) {
            read_samples.push_back(milliseconds(read_start, read_stop));
            preprocess_samples.push_back(frame_preprocess_ms);
            inference_samples.push_back(frame_inference_ms);
            decode_samples.push_back(frame_decode_ms);
            bev_samples.push_back(frame_bev_ms);
            core_samples.push_back(frame_core_ms);
            ++timed_frames;
        }
        ++frames;
        if (frames % 100U == 0U) {
            std::cerr << "processed " << frames << " frames\n";
        }
    }
    if (frames == 0 || timed_frames == 0) {
        throw std::runtime_error("raw stream produced no timed frames");
    }
    const auto read = summarize(std::move(read_samples));
    const auto preprocess = summarize(std::move(preprocess_samples));
    const auto inference = summarize(std::move(inference_samples));
    const auto decode = summarize(std::move(decode_samples));
    const auto bev = summarize(std::move(bev_samples));
    const auto core = summarize(std::move(core_samples));
    write_stream_report(
        args.report, frames, timed_frames, args.threads, read, preprocess,
        inference, decode, bev, core, lane_sum / static_cast<double>(frames)
    );
    std::cout << std::fixed << std::setprecision(3)
              << "C++ sequential raw-BEV benchmark (render/write excluded)\n"
              << "frames=" << frames << " timed=" << timed_frames
              << " threads=" << args.threads << '\n'
              << "preprocess median=" << preprocess.median << "ms\n"
              << "inference+D2H median=" << inference.median << "ms\n"
              << "decode median=" << decode.median << "ms\n"
              << "BEV median=" << bev.median << "ms\n"
              << "core median=" << core.median << "ms p95=" << core.p95
              << "ms FPS=" << (1000.0 / core.median) << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
        rclane::TensorRTRunner runner(args.engine);
        if (args.raw_bgr_stdin) {
            return run_raw_stream(args, runner);
        }
        std::vector<float> input;
        if (!args.input.empty()) {
            input = read_floats(args.input, runner.input_elements());
        } else {
            const auto bgr = read_bytes(
                args.input_bgr, static_cast<std::size_t>(1920 * 1080 * 3)
            );
            rclane::normalize_bgr_to_nchw(bgr.data(), 1920, 1080, input);
        }
        dump_input(args.dump_prefix, input);
        const auto outputs = runner.infer(input.data());
        dump_outputs(args.dump_prefix, outputs);
        rclane::DecoderConfig decoder_config;
        decoder_config.threads = args.threads;
        rclane::DecodeStatistics decode_statistics;
        const auto lanes = rclane::decode_outputs(
            outputs, decoder_config, &decode_statistics
        );
        if (!args.lanes_json.empty()) {
            rclane::write_lanes_json(
                args.lanes_json, lanes, &decode_statistics
            );
        }
        const auto bev_lanes = rclane::project_lanes_to_bev(lanes);
        if (!args.bev_json.empty()) {
            rclane::write_bev_json(args.bev_json, bev_lanes);
        }
        const auto timing = runner.benchmark(
            input.data(), args.warmup, args.iterations
        );
        std::cout << std::fixed << std::setprecision(3)
                  << "TensorRT C++ inference: mean=" << timing.mean_ms
                  << "ms median=" << timing.median_ms
                  << "ms p95=" << timing.p95_ms
                  << "ms min=" << timing.min_ms
                  << "ms max=" << timing.max_ms
                  << "ms FPS=" << (1000.0 / timing.median_ms) << '\n';
        for (const auto& [name, tensor] : outputs) {
            std::cout << name << " elements=" << tensor.values.size() << '\n';
        }
        std::cout << "decode seeds=" << decode_statistics.seeds
                  << " candidates=" << decode_statistics.crawled_candidates
                  << " NMS=" << decode_statistics.nms_candidates
                  << "->" << decode_statistics.nms_survivors
                  << " output_lanes=" << lanes.size() << '\n';
        std::cout << "BEV lanes=" << bev_lanes.size() << " valid_cubics="
                  << std::count_if(
                      bev_lanes.begin(), bev_lanes.end(),
                      [](const rclane::BevLane& lane) {
                          return lane.fit_accepted;
                      }
                  ) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
