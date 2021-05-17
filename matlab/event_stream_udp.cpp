#include "../sepia.hpp"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <sstream>

class MexFunction : public matlab::mex::Function {
    public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        matlab::data::ArrayFactory factory;
        try {
            if (inputs.size() != 1) {
                throw std::runtime_error("a single input argument is expected");
            }
            if (inputs[0].getType() != matlab::data::ArrayType::DOUBLE) {
                throw std::runtime_error("the first input argument must be a 1xN double array");
            }
            if (inputs[0].getDimensions().size() != 2 || inputs[0].getDimensions().front() != 1) {
                throw std::runtime_error("the first input argument must be a 1xN double array");
            }
            if (outputs.size() != 2) {
                throw std::runtime_error("two output arguments are expected");
            }
            if (inputs[0].getDimensions().back() < 8) {
                auto matlab_header = factory.createStructArray({1}, {"type"});
                matlab_header[0]["type"] = factory.createCharArray("generic");
                outputs[0] = std::move(matlab_header);
                auto matlab_events = factory.createStructArray({1}, {"t", "bytes"});
                matlab_events[0]["t"] = factory.createArray<uint64_t>({0, 1});
                matlab_events[0]["bytes"] = factory.createArray<matlab::data::MATLABString>({0, 1});
                outputs[1] = std::move(matlab_events);
            } else {
                const matlab::data::TypedArray<double> input_array = inputs[0];
                std::array<uint8_t, 8> timestamp_bytes;
                std::transform(
                    input_array.begin(), std::next(input_array.begin(), 8), timestamp_bytes.begin(), [](double byte) {
                        return static_cast<uint8_t>(byte);
                    });
                const auto t0 = *reinterpret_cast<const uint64_t*>(timestamp_bytes.data());
                std::string header_bytes;
                if (inputs[0].getDimensions().back() < 8 + 20) {
                    header_bytes.resize(inputs[0].getDimensions().back() - 8);
                    std::transform(
                        std::next(input_array.begin(), 8), input_array.end(), header_bytes.begin(), [](double byte) {
                            return static_cast<uint8_t>(byte);
                        });
                } else {
                    header_bytes.resize(20);
                    std::transform(
                        std::next(input_array.begin(), 8),
                        std::next(input_array.begin(), 8 + 20),
                        header_bytes.begin(),
                        [](double byte) { return static_cast<uint8_t>(byte); });
                }
                std::stringstream header_stream(header_bytes);
                const auto header = sepia::read_header(header_stream);
                switch (header.event_stream_type) {
                    case sepia::type::generic: {
                        {
                            auto matlab_header = factory.createStructArray({1}, {"type"});
                            matlab_header[0]["type"] = factory.createCharArray("generic");
                            outputs[0] = std::move(matlab_header);
                        }
                        const auto events = sepia::bytes_to_events<sepia::type::generic>(
                            t0, header, std::next(input_array.begin(), 8 + 16), input_array.end());
                        auto t = factory.createArray<uint64_t>({events.size(), 1});
                        auto bytes = factory.createArray<matlab::data::MATLABString>({events.size(), 1});
                        for (std::size_t index = 0; index < events.size(); ++index) {
                            t[index] = events[index].t;
                            bytes[index] = std::string(events[index].bytes.begin(), events[index].bytes.end());
                        }
                        auto matlab_events = factory.createStructArray({1}, {"t", "bytes"});
                        matlab_events[0]["t"] = std::move(t);
                        matlab_events[0]["bytes"] = std::move(bytes);
                        outputs[1] = std::move(matlab_events);
                        break;
                    }
                    case sepia::type::dvs: {
                        {
                            auto matlab_header = factory.createStructArray({1}, {"type", "width", "height"});
                            matlab_header[0]["type"] = factory.createCharArray("dvs");
                            matlab_header[0]["width"] = factory.createArray<uint16_t>({1, 1}, {header.width});
                            matlab_header[0]["height"] = factory.createArray<uint16_t>({1, 1}, {header.height});
                            outputs[0] = std::move(matlab_header);
                        }
                        const auto events = sepia::bytes_to_events<sepia::type::dvs>(
                            t0, header, std::next(input_array.begin(), 8 + 20), input_array.end());
                        auto t = factory.createArray<uint64_t>({events.size(), 1});
                        auto x = factory.createArray<uint16_t>({events.size(), 1});
                        auto y = factory.createArray<uint16_t>({events.size(), 1});
                        auto on = factory.createArray<bool>({events.size(), 1});
                        for (std::size_t index = 0; index < events.size(); ++index) {
                            t[index] = events[index].t;
                            x[index] = events[index].x;
                            y[index] = events[index].y;
                            on[index] = events[index].on;
                        }
                        auto matlab_events = factory.createStructArray({1}, {"t", "x", "y", "on"});
                        matlab_events[0]["t"] = std::move(t);
                        matlab_events[0]["x"] = std::move(x);
                        matlab_events[0]["y"] = std::move(y);
                        matlab_events[0]["on"] = std::move(on);
                        outputs[1] = std::move(matlab_events);
                        break;
                    }
                    case sepia::type::atis: {
                        {
                            auto matlab_header = factory.createStructArray({1}, {"type", "width", "height"});
                            matlab_header[0]["type"] = factory.createCharArray("atis");
                            matlab_header[0]["width"] = factory.createArray<uint16_t>({1, 1}, {header.width});
                            matlab_header[0]["height"] = factory.createArray<uint16_t>({1, 1}, {header.height});
                            outputs[0] = std::move(matlab_header);
                        }
                        const auto events = sepia::bytes_to_events<sepia::type::atis>(
                            t0, header, std::next(input_array.begin(), 8 + 20), input_array.end());
                        auto t = factory.createArray<uint64_t>({events.size(), 1});
                        auto x = factory.createArray<uint16_t>({events.size(), 1});
                        auto y = factory.createArray<uint16_t>({events.size(), 1});
                        auto exposure = factory.createArray<bool>({events.size(), 1});
                        auto polarity = factory.createArray<bool>({events.size(), 1});
                        for (std::size_t index = 0; index < events.size(); ++index) {
                            t[index] = events[index].t;
                            x[index] = events[index].x;
                            y[index] = events[index].y;
                            exposure[index] = events[index].exposure;
                            polarity[index] = events[index].polarity;
                        }
                        auto matlab_events = factory.createStructArray({1}, {"t", "x", "y", "exposure", "polarity"});
                        matlab_events[0]["t"] = std::move(t);
                        matlab_events[0]["x"] = std::move(x);
                        matlab_events[0]["y"] = std::move(y);
                        matlab_events[0]["exposure"] = std::move(exposure);
                        matlab_events[0]["polarity"] = std::move(polarity);
                        outputs[1] = std::move(matlab_events);
                        break;
                    }
                    case sepia::type::color: {
                        {
                            auto matlab_header = factory.createStructArray({1}, {"type", "width", "height"});
                            matlab_header[0]["type"] = factory.createCharArray("color");
                            matlab_header[0]["width"] = factory.createArray<uint16_t>({1, 1}, {header.width});
                            matlab_header[0]["height"] = factory.createArray<uint16_t>({1, 1}, {header.height});
                            outputs[0] = std::move(matlab_header);
                        }
                        const auto events = sepia::bytes_to_events<sepia::type::color>(
                            t0, header, std::next(input_array.begin(), 8 + 20), input_array.end());
                        auto t = factory.createArray<uint64_t>({events.size(), 1});
                        auto x = factory.createArray<uint16_t>({events.size(), 1});
                        auto y = factory.createArray<uint16_t>({events.size(), 1});
                        auto r = factory.createArray<uint8_t>({events.size(), 1});
                        auto g = factory.createArray<uint8_t>({events.size(), 1});
                        auto b = factory.createArray<uint8_t>({events.size(), 1});
                        for (std::size_t index = 0; index < events.size(); ++index) {
                            t[index] = events[index].t;
                            x[index] = events[index].x;
                            y[index] = events[index].y;
                            r[index] = events[index].r;
                            g[index] = events[index].g;
                            b[index] = events[index].b;
                        }
                        auto matlab_events = factory.createStructArray({1}, {"t", "x", "y", "r", "g", "b"});
                        matlab_events[0]["t"] = std::move(t);
                        matlab_events[0]["x"] = std::move(x);
                        matlab_events[0]["y"] = std::move(y);
                        matlab_events[0]["r"] = std::move(r);
                        matlab_events[0]["g"] = std::move(g);
                        matlab_events[0]["b"] = std::move(b);
                        outputs[1] = std::move(matlab_events);
                        break;
                    }
                    default:
                        throw std::runtime_error("unsupported event type");
                }
            }
        } catch (const std::exception& exception) {
            getEngine()->feval(
                matlab::engine::convertUTF8StringToUTF16String("error"),
                0,
                std::vector<matlab::data::Array>({factory.createScalar(exception.what())}));
        }
    }
};
