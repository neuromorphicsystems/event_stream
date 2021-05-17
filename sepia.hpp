#pragma once

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef SEPIA_COMPILER_WORKING_DIRECTORY
#ifdef _WIN32
std::string SEPIA_TRANSLATE(std::string filename) {
    for (auto& character : filename) {
        if (character == '/') {
            character = '\\';
        }
    }
    return filename;
}
#define SEPIA_DIRNAME                                                                                                  \
    sepia::dirname(                                                                                                    \
        std::string(__FILE__).size() > 1 && __FILE__[1] == ':' ?                                                       \
            __FILE__ :                                                                                                 \
            SEPIA_TRANSLATE(SEPIA_COMPILER_WORKING_DIRECTORY) + ("\\" __FILE__))
#else
#define SEPIA_STRINGIFY(characters) #characters
#define SEPIA_TOSTRING(characters) SEPIA_STRINGIFY(characters)
#define SEPIA_DIRNAME                                                                                                  \
    sepia::dirname(__FILE__[0] == '/' ? __FILE__ : SEPIA_TOSTRING(SEPIA_COMPILER_WORKING_DIRECTORY) "/" __FILE__)
#endif
#endif
#ifdef _WIN32
#define SEPIA_PACK(declaration) declaration
#else
#define SEPIA_PACK(declaration) declaration __attribute__((__packed__))
#endif

/// sepia bundles functions and classes to represent a camera and handle its raw
/// stream of events.
namespace sepia {

    /// event_stream_version returns the implemented Event Stream version.
    inline std::array<uint8_t, 3> event_stream_version() {
        return {2, 0, 0};
    }

    /// event_stream_signature returns the Event Stream format signature.
    inline std::string event_stream_signature() {
        return "Event Stream";
    }

    /// type associates an Event Stream type name with its byte.
    enum class type : uint8_t {
        generic = 0,
        dvs = 1,
        atis = 2,
        color = 4,
    };

    /// make_unique creates a unique_ptr.
    template <typename T, typename... Args>
    inline std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    /// false_function is a function returning false.
    inline bool false_function() {
        return false;
    }

    /// event represents the parameters of an observable event.
    template <type event_stream_type>
    struct event;

    /// generic_event represents the parameters of a generic event.
    template <>
    struct event<type::generic> {
        /// t represents the event's timestamp.
        uint64_t t;

        /// bytes stores the data payload associated with the event.
        std::vector<uint8_t> bytes;
    };
    using generic_event = event<type::generic>;

    /// dvs_event represents the parameters of a change detection.
    template <>
    SEPIA_PACK(struct event<type::dvs> {
        /// t represents the event's timestamp.
        uint64_t t;

        /// x represents the coordinate of the event on the sensor grid alongside the
        /// horizontal axis. x is 0 on the left, and increases from left to right.
        uint16_t x;

        /// y represents the coordinate of the event on the sensor grid alongside the
        /// vertical axis. y is 0 on the bottom, and increases from bottom to top.
        uint16_t y;

        /// on is false if the luminance decreased.
        bool on;
    });
    using dvs_event = event<type::dvs>;

    /// atis_event represents the parameters of a change detection or an exposure
    /// measurement.
    template <>
    SEPIA_PACK(struct event<type::atis> {
        /// t represents the event's timestamp.
        uint64_t t;

        /// x represents the coordinate of the event on the sensor grid alongside the
        /// horizontal axis. x is 0 on the left, and increases from left to right.
        uint16_t x;

        /// y represents the coordinate of the event on the sensor grid alongside the
        /// vertical axis. y is 0 on the bottom, and increases bottom to top.
        uint16_t y;

        /// exposure is false if the event is a change detection, and
        /// true if it is an exposure measurement.
        bool exposure;

        /// change detection: polarity is false if the light is decreasing.
        /// exposure measurement: polarity is false for a first threshold crossing.
        bool polarity;
    });
    using atis_event = event<type::atis>;

    /// color_event represents the parameters of a color event.
    template <>
    SEPIA_PACK(struct event<type::color> {
        /// t represents the event's timestamp.
        uint64_t t;

        /// x represents the coordinate of the event on the sensor grid alongside the
        /// horizontal axis. x is 0 on the left, and increases from left to right.
        uint16_t x;

        /// y represents the coordinate of the event on the sensor grid alongside the
        /// vertical axis. y is 0 on the bottom, and increases bottom to top.
        uint16_t y;

        /// r represents the red component of the color.
        uint8_t r;

        /// g represents the green component of the color.
        uint8_t g;

        /// b represents the blue component of the color.
        uint8_t b;
    });
    using color_event = event<type::color>;

    /// simple_event represents the parameters of a specialized DVS event.
    SEPIA_PACK(struct simple_event {
        /// t represents the event's timestamp.
        uint64_t t;

        /// x represents the coordinate of the event on the sensor grid alongside the
        /// horizontal axis. x is 0 on the left, and increases from left to right.
        uint16_t x;

        /// y represents the coordinate of the event on the sensor grid alongside the
        /// vertical axis. y is 0 on the bottom, and increases from bottom to top.
        uint16_t y;
    });

    /// exposure represents the parameters of a specialized ATIS event.
    SEPIA_PACK(struct exposure {
        /// t represents the event's timestamp.
        uint64_t t;

        /// x represents the coordinate of the event on the sensor grid alongside the
        /// horizontal axis. x is 0 on the left, and increases from left to right.
        uint16_t x;

        /// y represents the coordinate of the event on the sensor grid alongside the
        /// vertical axis. y is 0 on the bottom, and increases from bottom to top.
        uint16_t y;

        /// second is false if the event is a first threshold crossing.
        bool second;
    });

    /// unreadable_file is thrown when an input file does not exist or is not
    /// readable.
    class unreadable_file : public std::runtime_error {
        public:
        unreadable_file(const std::string& filename) :
            std::runtime_error("the file '" + filename + "' could not be open for reading") {}
    };

    /// unwritable_file is thrown whenan output file is not writable.
    class unwritable_file : public std::runtime_error {
        public:
        unwritable_file(const std::string& filename) :
            std::runtime_error("the file '" + filename + "'' could not be open for writing") {}
    };

    /// wrong_signature is thrown when an input file does not have the expected
    /// signature.
    class wrong_signature : public std::runtime_error {
        public:
        wrong_signature() : std::runtime_error("the stream does not have the expected signature") {}
    };

    /// unsupported_version is thrown when an Event Stream file uses an unsupported
    /// version.
    class unsupported_version : public std::runtime_error {
        public:
        unsupported_version() : std::runtime_error("the stream uses an unsupported version") {}
    };

    /// incomplete_header is thrown when the end of file is reached while reading
    /// the header.
    class incomplete_header : public std::runtime_error {
        public:
        incomplete_header() : std::runtime_error("the stream has an incomplete header") {}
    };

    /// unsupported_event_type is thrown when an Event Stream file uses an
    /// unsupported event type.
    class unsupported_event_type : public std::runtime_error {
        public:
        unsupported_event_type() : std::runtime_error("the stream uses an unsupported event type") {}
    };

    /// coordinates_overflow is thrown when an event has coordinates outside the
    /// range provided by the header.
    class coordinates_overflow : public std::runtime_error {
        public:
        coordinates_overflow() : std::runtime_error("an event has coordinates outside the header-provided range") {}
    };

    /// end_of_file is thrown when the end of an input file is reached.
    class end_of_file : public std::runtime_error {
        public:
        end_of_file() : std::runtime_error("end of file reached") {}
    };

    /// no_device_connected is thrown when device auto-select is called without
    /// devices connected.
    class no_device_connected : public std::runtime_error {
        public:
        no_device_connected(const std::string& device_family) :
            std::runtime_error("no " + device_family + " is connected") {}
    };

    /// device_disconnected is thrown when an active device is disonnected.
    class device_disconnected : public std::runtime_error {
        public:
        device_disconnected(const std::string& device_name) : std::runtime_error(device_name + " disconnected") {}
    };

    /// parse_error is thrown when a JSON parse error occurs.
    class parse_error : public std::runtime_error {
        public:
        parse_error(const std::string& what, std::size_t character_count, std::size_t line_count) :
            std::runtime_error(
                "JSON parse error: " + what + " (line " + std::to_string(line_count) + ":"
                + std::to_string(character_count) + ")") {}
    };

    /// parameter_error is a logical error regarding a parameter.
    class parameter_error : public std::logic_error {
        public:
        parameter_error(const std::string& what) : std::logic_error(what) {}
    };

    /// dirname returns the directory part of the given path.
    inline std::string dirname(const std::string& path) {
#ifdef _WIN32
        const auto separator = '\\';
        const auto escape = '^';
#else
        const auto separator = '/';
        const auto escape = '\\';
#endif
        for (std::size_t index = path.size();;) {
            index = path.find_last_of(separator, index);
            if (index == std::string::npos) {
                return ".";
            }
            if (index == 0 || path[index - 1] != escape) {
                return path.substr(0, index);
            }
        }
    }

    /// join concatenates several path components.
    template <typename Iterator>
    inline std::string join(Iterator begin, Iterator end) {
#ifdef _WIN32
        const auto separator = '\\';
#else
        const auto separator = '/';
#endif
        std::string path;
        for (; begin != end; ++begin) {
            path += *begin;
            if (!path.empty() && begin != std::prev(end) && path.back() != separator) {
                path.push_back(separator);
            }
        }
        return path;
    }
    inline std::string join(std::initializer_list<std::string> components) {
        return join(components.begin(), components.end());
    }

    /// filename_to_ifstream creates a readable stream from a file.
    inline std::unique_ptr<std::ifstream> filename_to_ifstream(const std::string& filename) {
        auto stream = sepia::make_unique<std::ifstream>(filename, std::ifstream::in | std::ifstream::binary);
        if (!stream->good()) {
            throw unreadable_file(filename);
        }
        return stream;
    }

    /// filename_to_ofstream creates a writable stream from a file.
    inline std::unique_ptr<std::ofstream> filename_to_ofstream(const std::string& filename) {
        auto stream = sepia::make_unique<std::ofstream>(filename, std::ofstream::out | std::ofstream::binary);
        if (!stream->good()) {
            throw unwritable_file(filename);
        }
        return stream;
    }

    /// header bundles an event stream's header parameters.
    struct header {
        /// version contains the version's major, minor and patch numbers in that
        /// order.
        std::array<uint8_t, 3> version;

        /// event_stream_type is the type of the events in the associated stream.
        type event_stream_type;

        /// width is at least one more than the largest x coordinate among the
        /// stream's events.
        uint16_t width;

        /// heaight is at least one more than the largest y coordinate among the
        /// stream's events.
        uint16_t height;
    };

    /// read_header checks the header and retrieves meta-information from the given
    /// stream.
    inline header read_header(std::istream& event_stream) {
        {
            auto read_signature = event_stream_signature();
            event_stream.read(&read_signature[0], read_signature.size());
            if (event_stream.eof() || read_signature != event_stream_signature()) {
                throw wrong_signature();
            }
        }
        header header = {};
        {
            event_stream.read(reinterpret_cast<char*>(header.version.data()), header.version.size());
            if (event_stream.eof()) {
                throw incomplete_header();
            }
            if (std::get<0>(header.version) != std::get<0>(event_stream_version())
                || std::get<1>(header.version) < std::get<1>(event_stream_version())) {
                throw unsupported_version();
            }
        }
        {
            const auto type_char = static_cast<char>(event_stream.get());
            if (event_stream.eof()) {
                throw incomplete_header();
            }
            const auto type_byte = *reinterpret_cast<const uint8_t*>(&type_char);
            if (type_byte == static_cast<uint8_t>(type::generic)) {
                header.event_stream_type = type::generic;
            } else if (type_byte == static_cast<uint8_t>(type::dvs)) {
                header.event_stream_type = type::dvs;
            } else if (type_byte == static_cast<uint8_t>(type::atis)) {
                header.event_stream_type = type::atis;
            } else if (type_byte == static_cast<uint8_t>(type::color)) {
                header.event_stream_type = type::color;
            } else {
                throw unsupported_event_type();
            }
        }
        if (header.event_stream_type != type::generic) {
            std::array<uint8_t, 4> size_bytes;
            event_stream.read(reinterpret_cast<char*>(size_bytes.data()), size_bytes.size());
            if (event_stream.eof()) {
                throw incomplete_header();
            }
            header.width = static_cast<uint16_t>(
                (static_cast<uint16_t>(std::get<0>(size_bytes))
                 | (static_cast<uint16_t>(std::get<1>(size_bytes)) << 8)));
            header.height = static_cast<uint16_t>(
                (static_cast<uint16_t>(std::get<2>(size_bytes))
                 | (static_cast<uint16_t>(std::get<3>(size_bytes)) << 8)));
        }
        return header;
    }

    /// read_header checks the header and retrieves meta-information from the given
    /// stream.
    inline header read_header(std::unique_ptr<std::istream> event_stream) {
        return read_header(*event_stream);
    }

    /// write_header writes the header bytes to a byte stream.
    template <type event_stream_type>
    inline void write_header(std::ostream& event_stream, uint16_t width, uint16_t height) {
        event_stream.write(event_stream_signature().data(), event_stream_signature().size());
        event_stream.write(reinterpret_cast<char*>(event_stream_version().data()), event_stream_version().size());
        std::array<uint8_t, 5> bytes{
            static_cast<uint8_t>(event_stream_type),
            static_cast<uint8_t>(width & 0b11111111),
            static_cast<uint8_t>((width & 0b1111111100000000) >> 8),
            static_cast<uint8_t>(height & 0b11111111),
            static_cast<uint8_t>((height & 0b1111111100000000) >> 8),
        };
        event_stream.write(reinterpret_cast<char*>(bytes.data()), bytes.size());
    }
    template <type event_stream_type, typename = typename std::enable_if<event_stream_type == type::generic>>
    inline void write_header(std::ostream& event_stream) {
        event_stream.write(event_stream_signature().data(), event_stream_signature().size());
        event_stream.write(reinterpret_cast<char*>(event_stream_version().data()), event_stream_version().size());
        auto type_byte = static_cast<uint8_t>(event_stream_type);
        event_stream.put(*reinterpret_cast<char*>(&type_byte));
    }
    template <>
    inline void write_header<type::generic>(std::ostream& event_stream, uint16_t, uint16_t) {
        write_header<type::generic>(event_stream);
    }

    /// split separates a stream of DVS or ATIS events into specialized streams.
    template <type event_stream_type, typename HandleFirstSpecializedEvent, typename HandleSecondSpecializedEvent>
    class split;

    /// split separates a stream of DVS events into two streams of simple events.
    template <typename HandleIncreaseEvent, typename HandleDecreaseEvent>
    class split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent> {
        public:
        split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>(
            HandleIncreaseEvent&& handle_increase_event,
            HandleDecreaseEvent&& handle_decrease_event) :
            _handle_increase_event(std::forward<HandleIncreaseEvent>(handle_increase_event)),
            _handle_decrease_event(std::forward<HandleDecreaseEvent>(handle_decrease_event)) {}
        split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>(
            const split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>&) = delete;
        split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>(
            split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>&&) = default;
        split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>&
        operator=(const split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>&) = delete;
        split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>&
        operator=(split<type::dvs, HandleIncreaseEvent, HandleDecreaseEvent>&&) = default;
        virtual ~split() {}

        /// operator() handles an event.
        void operator()(dvs_event dvs_event) {
            if (dvs_event.on) {
                _handle_increase_event(simple_event{dvs_event.t, dvs_event.x, dvs_event.y});
            } else {
                _handle_decrease_event(simple_event{dvs_event.t, dvs_event.x, dvs_event.y});
            }
        }

        protected:
        HandleIncreaseEvent _handle_increase_event;
        HandleDecreaseEvent _handle_decrease_event;
    };

    /// split separates a stream of ATIS events into a stream of DVS events and a
    /// stream of theshold crossings.
    template <typename HandleDvsEvent, typename HandleExposure>
    class split<type::atis, HandleDvsEvent, HandleExposure> {
        public:
        split<type::atis, HandleDvsEvent, HandleExposure>(
            HandleDvsEvent&& handle_dvs_event,
            HandleExposure&& handle_exposure) :
            _handle_dvs_event(std::forward<HandleDvsEvent>(handle_dvs_event)),
            _handle_exposure(std::forward<HandleExposure>(handle_exposure)) {}
        split<type::atis, HandleDvsEvent, HandleExposure>(const split<type::atis, HandleDvsEvent, HandleExposure>&) =
            delete;
        split<type::atis, HandleDvsEvent, HandleExposure>(split<type::atis, HandleDvsEvent, HandleExposure>&&) =
            default;
        split<type::atis, HandleDvsEvent, HandleExposure>&
        operator=(const split<type::atis, HandleDvsEvent, HandleExposure>&) = delete;
        split<type::atis, HandleDvsEvent, HandleExposure>&
        operator=(split<type::atis, HandleDvsEvent, HandleExposure>&&) = default;
        virtual ~split<type::atis, HandleDvsEvent, HandleExposure>() {}

        /// operator() handles an event.
        void operator()(atis_event current_atis_event) {
            if (current_atis_event.exposure) {
                _handle_exposure(exposure{
                    current_atis_event.t, current_atis_event.x, current_atis_event.y, current_atis_event.polarity});
            } else {
                _handle_dvs_event(dvs_event{
                    current_atis_event.t, current_atis_event.x, current_atis_event.y, current_atis_event.polarity});
            }
        }

        protected:
        HandleDvsEvent _handle_dvs_event;
        HandleExposure _handle_exposure;
    };

    /// make_split creates a split from functors.
    template <type event_stream_type, typename HandleFirstSpecializedEvent, typename HandleSecondSpecializedEvent>
    inline split<event_stream_type, HandleFirstSpecializedEvent, HandleSecondSpecializedEvent> make_split(
        HandleFirstSpecializedEvent&& handle_first_specialized_event,
        HandleSecondSpecializedEvent&& handle_second_specialized_event) {
        return split<event_stream_type, HandleFirstSpecializedEvent, HandleSecondSpecializedEvent>(
            std::forward<HandleFirstSpecializedEvent>(handle_first_specialized_event),
            std::forward<HandleSecondSpecializedEvent>(handle_second_specialized_event));
    }

    /// event_size returns the number of bytes required to encode an event.
    template <type event_stream_type>
    uint8_t event_size(event<event_stream_type>);

    /// keyframe associates a number of bytes and a timestamp.
    struct keyframe {
        std::size_t offset;
        uint64_t t;
    };

    /// handle_byte implements an event stream state machine.
    template <type event_stream_type>
    class handle_byte;

    /// handle_byte<type::generic> implements the event stream state machine for
    /// generic events.
    template <>
    class handle_byte<type::generic> {
        public:
        handle_byte() : _state(state::idle), _index(0), _bytes_size(0), _relative_keyframe(keyframe{0, 0}) {}
        handle_byte(uint16_t, uint16_t) : handle_byte() {}
        handle_byte(const handle_byte&) = default;
        handle_byte(handle_byte&&) = default;
        handle_byte& operator=(const handle_byte&) = default;
        handle_byte& operator=(handle_byte&&) = default;
        virtual ~handle_byte() {}

        /// operator() handles a byte.
        bool operator()(uint8_t byte, generic_event& generic_event) {
            ++_relative_keyframe.offset;
            switch (_state) {
                case state::idle:
                    if (byte == 0b11111111) {
                        generic_event.t += 0b11111110;
                        _relative_keyframe.offset = 0;
                        _relative_keyframe.t = generic_event.t;
                    } else if (byte != 0b11111110) {
                        _relative_keyframe.offset = 1;
                        _relative_keyframe.t = generic_event.t;
                        generic_event.t += byte;
                        _state = state::byte0;
                    } else {
                        _relative_keyframe.offset = 0;
                        _relative_keyframe.t = generic_event.t;
                    }
                    break;
                case state::byte0:
                    _bytes_size |= ((byte >> 1) << (7 * _index));
                    if ((byte & 1) == 0) {
                        generic_event.bytes.clear();
                        _index = 0;
                        if (_bytes_size == 0) {
                            _state = state::idle;
                            return true;
                        }
                        generic_event.bytes.reserve(_bytes_size);
                        _state = state::size_byte;
                    } else {
                        ++_index;
                    }
                    break;
                case state::size_byte:
                    generic_event.bytes.push_back(byte);
                    if (generic_event.bytes.size() == _bytes_size) {
                        _state = state::idle;
                        _index = 0;
                        _bytes_size = 0;
                        return true;
                    }
                    break;
            }
            return false;
        }

        /// reset initializes the state machine.
        void reset() {
            _state = state::idle;
            _index = 0;
            _bytes_size = 0;
        }

        /// relative_keyframe returns the number of bytes consummed since the last event boundary,
        /// and the timestamp of the last event.
        keyframe relative_keyframe() const {
            return _relative_keyframe;
        }

        protected:
        /// state represents the current state machine's state.
        enum class state {
            idle,
            byte0,
            size_byte,
        };

        state _state;
        std::size_t _index;
        std::size_t _bytes_size;
        keyframe _relative_keyframe;
    };

    /// handle_byte<type::dvs> implements the event stream state machine for DVS
    /// events.
    template <>
    class handle_byte<type::dvs> {
        public:
        handle_byte(uint16_t width, uint16_t height) :
            _width(width), _height(height), _state(state::idle), _relative_keyframe(keyframe{0, 0}) {}
        handle_byte(const handle_byte&) = default;
        handle_byte(handle_byte&&) = default;
        handle_byte& operator=(const handle_byte&) = default;
        handle_byte& operator=(handle_byte&&) = default;
        virtual ~handle_byte() {}

        /// operator() handles a byte.
        bool operator()(uint8_t byte, dvs_event& dvs_event) {
            ++_relative_keyframe.offset;
            switch (_state) {
                case state::idle:
                    if (byte == 0b11111111) {
                        dvs_event.t += 0b1111111;
                        _relative_keyframe.offset = 0;
                        _relative_keyframe.t = dvs_event.t;
                    } else if (byte != 0b11111110) {
                        _relative_keyframe.offset = 1;
                        _relative_keyframe.t = dvs_event.t;
                        dvs_event.t += (byte >> 1);
                        dvs_event.on = ((byte & 1) == 1);
                        _state = state::byte0;
                    } else {
                        _relative_keyframe.offset = 0;
                        _relative_keyframe.t = dvs_event.t;
                    }
                    break;
                case state::byte0:
                    dvs_event.x = byte;
                    _state = state::byte1;
                    break;
                case state::byte1:
                    dvs_event.x |= (byte << 8);
                    if (dvs_event.x >= _width) {
                        throw coordinates_overflow();
                    }
                    _state = state::byte2;
                    break;
                case state::byte2:
                    dvs_event.y = byte;
                    _state = state::byte3;
                    break;
                case state::byte3:
                    dvs_event.y |= (byte << 8);
                    if (dvs_event.y >= _height) {
                        throw coordinates_overflow();
                    }
                    _state = state::idle;
                    return true;
            }
            return false;
        }

        /// reset initializes the state machine.
        void reset() {
            _state = state::idle;
        }

        /// relative_keyframe returns the number of bytes consummed since the last event boundary,
        /// and the timestamp of the last event.
        keyframe relative_keyframe() const {
            return _relative_keyframe;
        }

        protected:
        /// state represents the current state machine's state.
        enum class state {
            idle,
            byte0,
            byte1,
            byte2,
            byte3,
        };

        uint16_t _width;
        uint16_t _height;
        state _state;
        keyframe _relative_keyframe;
    };

    /// handle_byte<type::atis> implements the event stream state machine for ATIS
    /// events.
    template <>
    class handle_byte<type::atis> {
        public:
        handle_byte(uint16_t width, uint16_t height) :
            _width(width), _height(height), _state(state::idle), _relative_keyframe(keyframe{0, 0}) {}
        handle_byte(const handle_byte&) = default;
        handle_byte(handle_byte&&) = default;
        handle_byte& operator=(const handle_byte&) = default;
        handle_byte& operator=(handle_byte&&) = default;
        virtual ~handle_byte() {}

        /// operator() handles a byte.
        bool operator()(uint8_t byte, atis_event& atis_event) {
            ++_relative_keyframe.offset;
            switch (_state) {
                case state::idle:
                    if ((byte & 0b11111100) == 0b11111100) {
                        atis_event.t += static_cast<uint64_t>(0b111111) * (byte & 0b11);
                        _relative_keyframe.offset = 0;
                        _relative_keyframe.t = atis_event.t;
                    } else {
                        _relative_keyframe.offset = 1;
                        _relative_keyframe.t = atis_event.t;
                        atis_event.t += (byte >> 2);
                        atis_event.exposure = ((byte & 1) == 1);
                        atis_event.polarity = ((byte & 0b10) == 0b10);
                        _state = state::byte0;
                    }
                    break;
                case state::byte0:
                    atis_event.x = byte;
                    _state = state::byte1;
                    break;
                case state::byte1:
                    atis_event.x |= (byte << 8);
                    if (atis_event.x >= _width) {
                        throw coordinates_overflow();
                    }
                    _state = state::byte2;
                    break;
                case state::byte2:
                    atis_event.y = byte;
                    _state = state::byte3;
                    break;
                case state::byte3:
                    atis_event.y |= (byte << 8);
                    if (atis_event.y >= _height) {
                        throw coordinates_overflow();
                    }
                    _state = state::idle;
                    return true;
            }
            return false;
        }

        /// reset initializes the state machine.
        void reset() {
            _state = state::idle;
        }

        /// relative_keyframe returns the number of bytes consummed since the last event boundary,
        /// and the timestamp of the last event.
        keyframe relative_keyframe() const {
            return _relative_keyframe;
        }

        protected:
        /// state represents the current state machine's state.
        enum class state {
            idle,
            byte0,
            byte1,
            byte2,
            byte3,
        };

        uint16_t _width;
        uint16_t _height;
        state _state;
        keyframe _relative_keyframe;
    };

    /// handle_byte<type::color> implements the event stream state machine for color
    /// events.
    template <>
    class handle_byte<type::color> {
        public:
        handle_byte() = default;
        handle_byte(uint16_t width, uint16_t height) :
            _width(width), _height(height), _state(state::idle), _relative_keyframe(keyframe{0, 0}) {}
        handle_byte(const handle_byte&) = default;
        handle_byte(handle_byte&&) = default;
        handle_byte& operator=(const handle_byte&) = default;
        handle_byte& operator=(handle_byte&&) = default;
        virtual ~handle_byte() {}

        /// operator() handles a byte.
        bool operator()(uint8_t byte, color_event& color_event) {
            ++_relative_keyframe.offset;
            switch (_state) {
                case state::idle: {
                    if (byte == 0b11111111) {
                        color_event.t += 0b11111110;
                        _relative_keyframe.offset = 0;
                        _relative_keyframe.t = color_event.t;
                    } else if (byte != 0b11111110) {
                        _relative_keyframe.offset = 1;
                        _relative_keyframe.t = color_event.t;
                        color_event.t += byte;
                        _state = state::byte0;
                    } else {
                        _relative_keyframe.offset = 0;
                        _relative_keyframe.t = color_event.t;
                    }
                    break;
                }
                case state::byte0: {
                    color_event.x = byte;
                    _state = state::byte1;
                    break;
                }
                case state::byte1: {
                    color_event.x |= (byte << 8);
                    if (color_event.x >= _width) {
                        throw coordinates_overflow();
                    }
                    _state = state::byte2;
                    break;
                }
                case state::byte2: {
                    color_event.y = byte;
                    _state = state::byte3;
                    break;
                }
                case state::byte3: {
                    color_event.y |= (byte << 8);
                    if (color_event.y >= _height) {
                        throw coordinates_overflow();
                    }
                    _state = state::byte4;
                    break;
                }
                case state::byte4: {
                    color_event.r = byte;
                    _state = state::byte5;
                    break;
                }
                case state::byte5: {
                    color_event.g = byte;
                    _state = state::byte6;
                    break;
                }
                case state::byte6: {
                    color_event.b = byte;
                    _state = state::idle;
                    return true;
                }
            }
            return false;
        }

        /// reset initializes the state machine.
        void reset() {
            _state = state::idle;
        }

        /// relative_keyframe returns the number of bytes consummed since the last event boundary,
        /// and the timestamp of the last event.
        keyframe relative_keyframe() const {
            return _relative_keyframe;
        }

        protected:
        /// state represents the current state machine's state.
        enum class state {
            idle,
            byte0,
            byte1,
            byte2,
            byte3,
            byte4,
            byte5,
            byte6,
        };

        uint16_t _width;
        uint16_t _height;
        state _state;
        keyframe _relative_keyframe;
    };

    /// write_to_reference converts and writes events to a non-owned byte stream.
    template <type event_stream_type>
    class write_to_reference;

    /// write_to_reference<type::generic> converts and writes generic events to a
    /// non-owned byte stream.
    template <>
    class write_to_reference<type::generic> {
        public:
        write_to_reference(std::ostream& event_stream) : _event_stream(event_stream), _previous_t(0) {
            write_header<type::generic>(_event_stream);
        }
        write_to_reference(std::ostream& event_stream, uint16_t, uint16_t) : write_to_reference(event_stream) {}
        write_to_reference(const write_to_reference&) = delete;
        write_to_reference(write_to_reference&&) = default;
        write_to_reference& operator=(const write_to_reference&) = delete;
        write_to_reference& operator=(write_to_reference&&) = delete;
        virtual ~write_to_reference() {}

        /// operator() handles an event.
        void operator()(generic_event current_generic_event) {
            if (current_generic_event.t < _previous_t) {
                throw std::logic_error("the event's timestamp is smaller than the previous one's");
            }
            auto relative_t = current_generic_event.t - _previous_t;
            if (relative_t >= 0b11111110) {
                const auto number_of_overflows = relative_t / 0b11111110;
                for (std::size_t index = 0; index < number_of_overflows; ++index) {
                    _event_stream.put(static_cast<uint8_t>(0b11111111));
                }
                relative_t -= number_of_overflows * 0b11111110;
            }
            _event_stream.put(static_cast<uint8_t>(relative_t));
            for (std::size_t size = current_generic_event.bytes.size(); size > 0; size >>= 7) {
                _event_stream.put(static_cast<uint8_t>((size & 0b1111111) << 1) | ((size >> 7) > 0 ? 1 : 0));
            }
            _event_stream.write(
                reinterpret_cast<const char*>(current_generic_event.bytes.data()), current_generic_event.bytes.size());
            _previous_t = current_generic_event.t;
        }

        protected:
        std::ostream& _event_stream;
        uint64_t _previous_t;
    };

    /// write_to_reference<type::dvs> converts and writes DVS events to a non-owned
    /// byte stream.
    template <>
    class write_to_reference<type::dvs> {
        public:
        write_to_reference(std::ostream& event_stream, uint16_t width, uint16_t height) :
            _event_stream(event_stream), _width(width), _height(height), _previous_t(0) {
            write_header<type::dvs>(_event_stream, width, height);
        }
        write_to_reference(const write_to_reference&) = delete;
        write_to_reference(write_to_reference&&) = default;
        write_to_reference& operator=(const write_to_reference&) = delete;
        write_to_reference& operator=(write_to_reference&&) = delete;
        virtual ~write_to_reference() {}

        /// operator() handles an event.
        void operator()(dvs_event current_dvs_event) {
            if (current_dvs_event.x >= _width || current_dvs_event.y >= _height) {
                throw coordinates_overflow();
            }
            if (current_dvs_event.t < _previous_t) {
                throw std::logic_error("the event's timestamp is smaller than the previous one's");
            }
            auto relative_t = current_dvs_event.t - _previous_t;
            if (relative_t >= 0b1111111) {
                const auto number_of_overflows = relative_t / 0b1111111;
                for (std::size_t index = 0; index < number_of_overflows; ++index) {
                    _event_stream.put(static_cast<uint8_t>(0b11111111));
                }
                relative_t -= number_of_overflows * 0b1111111;
            }
            std::array<uint8_t, 5> bytes{
                static_cast<uint8_t>((relative_t << 1) | (current_dvs_event.on ? 1 : 0)),
                static_cast<uint8_t>(current_dvs_event.x & 0b11111111),
                static_cast<uint8_t>((current_dvs_event.x & 0b1111111100000000) >> 8),
                static_cast<uint8_t>(current_dvs_event.y & 0b11111111),
                static_cast<uint8_t>((current_dvs_event.y & 0b1111111100000000) >> 8),
            };
            _event_stream.write(reinterpret_cast<char*>(bytes.data()), bytes.size());
            _previous_t = current_dvs_event.t;
        }

        protected:
        std::ostream& _event_stream;
        const uint16_t _width;
        const uint16_t _height;
        uint64_t _previous_t;
    };

    /// write_to_reference<type::atis> converts and writes ATIS events to a
    /// non-owned byte stream.
    template <>
    class write_to_reference<type::atis> {
        public:
        write_to_reference(std::ostream& event_stream, uint16_t width, uint16_t height) :
            _event_stream(event_stream), _width(width), _height(height), _previous_t(0) {
            write_header<type::atis>(_event_stream, width, height);
        }
        write_to_reference(const write_to_reference&) = delete;
        write_to_reference(write_to_reference&&) = default;
        write_to_reference& operator=(const write_to_reference&) = delete;
        write_to_reference& operator=(write_to_reference&&) = delete;
        virtual ~write_to_reference() {}

        /// operator() handles an event.
        void operator()(atis_event current_atis_event) {
            if (current_atis_event.x >= _width || current_atis_event.y >= _height) {
                throw coordinates_overflow();
            }
            if (current_atis_event.t < _previous_t) {
                throw std::logic_error("the event's timestamp is smaller than the previous one's");
            }
            auto relative_t = current_atis_event.t - _previous_t;
            if (relative_t >= 0b111111) {
                const auto number_of_overflows = relative_t / 0b111111;
                for (std::size_t index = 0; index < number_of_overflows / 0b11; ++index) {
                    _event_stream.put(static_cast<uint8_t>(0b11111111));
                }
                const auto number_of_overflows_left = number_of_overflows % 0b11;
                if (number_of_overflows_left > 0) {
                    _event_stream.put(static_cast<uint8_t>(0b11111100 | number_of_overflows_left));
                }
                relative_t -= number_of_overflows * 0b111111;
            }
            std::array<uint8_t, 5> bytes{
                static_cast<uint8_t>(
                    (relative_t << 2) | (current_atis_event.polarity ? 0b10 : 0b00)
                    | (current_atis_event.exposure ? 1 : 0)),
                static_cast<uint8_t>(current_atis_event.x & 0b11111111),
                static_cast<uint8_t>((current_atis_event.x & 0b1111111100000000) >> 8),
                static_cast<uint8_t>(current_atis_event.y & 0b11111111),
                static_cast<uint8_t>((current_atis_event.y & 0b1111111100000000) >> 8),
            };
            _event_stream.write(reinterpret_cast<char*>(bytes.data()), bytes.size());
            _previous_t = current_atis_event.t;
        }

        protected:
        std::ostream& _event_stream;
        const uint16_t _width;
        const uint16_t _height;
        uint64_t _previous_t;
    };

    /// write_to_reference<type::color> converts and writes color events to a
    /// non-owned byte stream.
    template <>
    class write_to_reference<type::color> {
        public:
        write_to_reference(std::ostream& event_stream, uint16_t width, uint16_t height) :
            _event_stream(event_stream), _width(width), _height(height), _previous_t(0) {
            write_header<type::color>(_event_stream, width, height);
        }
        write_to_reference(const write_to_reference&) = delete;
        write_to_reference(write_to_reference&&) = default;
        write_to_reference& operator=(const write_to_reference&) = delete;
        write_to_reference& operator=(write_to_reference&&) = delete;
        virtual ~write_to_reference() {}

        /// operator() handles an event.
        void operator()(color_event current_color_event) {
            if (current_color_event.x >= _width || current_color_event.y >= _height) {
                throw coordinates_overflow();
            }
            if (current_color_event.t < _previous_t) {
                throw std::logic_error("the event's timestamp is smaller than the previous one's");
            }
            auto relative_t = current_color_event.t - _previous_t;
            if (relative_t >= 0b11111110) {
                const auto number_of_overflows = relative_t / 0b11111110;
                for (std::size_t index = 0; index < number_of_overflows; ++index) {
                    _event_stream.put(static_cast<uint8_t>(0b11111111));
                }
                relative_t -= number_of_overflows * 0b11111110;
            }
            std::array<uint8_t, 8> bytes{
                static_cast<uint8_t>(relative_t),
                static_cast<uint8_t>(current_color_event.x & 0b11111111),
                static_cast<uint8_t>((current_color_event.x & 0b1111111100000000) >> 8),
                static_cast<uint8_t>(current_color_event.y & 0b11111111),
                static_cast<uint8_t>((current_color_event.y & 0b1111111100000000) >> 8),
                static_cast<uint8_t>(current_color_event.r),
                static_cast<uint8_t>(current_color_event.g),
                static_cast<uint8_t>(current_color_event.b),
            };
            _event_stream.write(reinterpret_cast<char*>(bytes.data()), bytes.size());
            _previous_t = current_color_event.t;
        }

        protected:
        std::ostream& _event_stream;
        const uint16_t _width;
        const uint16_t _height;
        uint64_t _previous_t;
    };

    class any_write {
        public:
        any_write() = default;
        any_write(const any_write&) = default;
        any_write(any_write&&) = default;
        any_write& operator=(const any_write&) = default;
        any_write& operator=(any_write&&) = default;
        virtual ~any_write() {}
    };

    /// write converts and writes events to a byte stream.
    template <type event_stream_type>
    class write : public any_write {
        public:
        template <type generic_type = type::generic>
        write(
            std::unique_ptr<std::ostream> event_stream,
            typename std::enable_if<event_stream_type == generic_type>::type* = nullptr) :
            write(std::move(event_stream), 0, 0) {}
        write(std::unique_ptr<std::ostream> event_stream, uint16_t width, uint16_t height) :
            _event_stream(std::move(event_stream)), _write_to_reference(*_event_stream, width, height) {}
        write(const write&) = delete;
        write(write&&) = default;
        write& operator=(const write&) = delete;
        write& operator=(write&&) = default;
        virtual ~write() {}

        /// operator() handles an event.
        void operator()(event<event_stream_type> event) {
            _write_to_reference(event);
        }

        protected:
        std::unique_ptr<std::ostream> _event_stream;
        write_to_reference<event_stream_type> _write_to_reference;
    };

    class any_observable {
        public:
        any_observable() = default;
        any_observable(const any_observable&) = default;
        any_observable(any_observable&&) = default;
        any_observable& operator=(const any_observable&) = default;
        any_observable& operator=(any_observable&&) = default;
        virtual ~any_observable() {}
    };

    /// observable reads bytes from a stream and dispatches events.
    template <type event_stream_type>
    class observable : public any_observable {
        public:
        observable(std::unique_ptr<std::istream> event_stream) :
            any_observable(), _event_stream(std::move(event_stream)), _handle_byte(0, 0) {
            const auto header = read_header(*_event_stream);
            if (header.event_stream_type != event_stream_type) {
                throw unsupported_event_type();
            }
            _event = event<event_stream_type>{};
            _handle_byte = handle_byte<event_stream_type>(header.width, header.height);
        }
        observable(const observable&) = delete;
        observable(observable&&) = default;
        observable& operator=(const observable&) = delete;
        observable& operator=(observable&&) = default;
        virtual ~observable() {}

        /// next yields a new event buffer.
        /// The chunk size is the number of bytes read, not the number of events.
        /// next always return a non-empty buffer until end-of-file is reached.
        const std::vector<sepia::event<event_stream_type>>& next(std::size_t chunk_size = 1 << 16) {
            _bytes.resize(chunk_size);
            _buffer.clear();
            _buffer.reserve(chunk_size);
            for (;;) {
                _event_stream->read(reinterpret_cast<char*>(_bytes.data()), _bytes.size());
                if (_event_stream->eof()) {
                    for (auto byte_iterator = _bytes.begin();
                         byte_iterator
                         != std::next(
                             _bytes.begin(),
                             static_cast<std::iterator_traits<std::vector<uint8_t>::iterator>::difference_type>(
                                 _event_stream->gcount()));
                         ++byte_iterator) {
                        if (_handle_byte(*byte_iterator, _event)) {
                            _buffer.push_back(_event);
                        }
                    }
                    break;
                } else {
                    for (auto byte : _bytes) {
                        if (_handle_byte(byte, _event)) {
                            _buffer.push_back(_event);
                        }
                    }
                    if (!_buffer.empty()) {
                        break;
                    }
                }
            }
            return _buffer;
        }

        protected:
        std::unique_ptr<std::istream> _event_stream;
        handle_byte<event_stream_type> _handle_byte;
        event<event_stream_type> _event;
        std::vector<uint8_t> _bytes;
        std::vector<sepia::event<event_stream_type>> _buffer;
    };

    /// make_observable creates a smart pointer to an event stream observable.
    template <type event_stream_type>
    inline std::unique_ptr<observable<event_stream_type>> make_observable(std::unique_ptr<std::istream> event_stream) {
        return sepia::make_unique<observable<event_stream_type>>(std::move(event_stream));
    }

    /// read_events reads all the events from the given stream.
    template <type event_stream_type, typename HandleEvent>
    inline void read_events(std::istream& event_stream, HandleEvent&& handle_event, std::size_t chunk_size = 1 << 16) {
        const auto header = read_header(event_stream);
        if (header.event_stream_type != event_stream_type) {
            throw unsupported_event_type();
        }
        auto stream_handle_byte = handle_byte<event_stream_type>(header.width, header.height);
        std::vector<uint8_t> bytes(chunk_size);
        event<event_stream_type> stream_event = {};
        for (;;) {
            event_stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
            if (event_stream.eof()) {
                for (auto byte_iterator = bytes.begin();
                     byte_iterator
                     != std::next(
                         bytes.begin(),
                         static_cast<std::iterator_traits<std::vector<uint8_t>::iterator>::difference_type>(
                             event_stream.gcount()));
                     ++byte_iterator) {
                    if (stream_handle_byte(*byte_iterator, stream_event)) {
                        handle_event(stream_event);
                    }
                }
                break;
            } else {
                for (auto byte : bytes) {
                    if (stream_handle_byte(byte, stream_event)) {
                        handle_event(stream_event);
                    }
                }
            }
        }
    }
    template <type event_stream_type, typename HandleEvent>
    inline void read_events(
        std::unique_ptr<std::istream> event_stream,
        HandleEvent&& handle_event,
        std::size_t chunk_size = 1 << 16) {
        return read_events<event_stream_type>(*event_stream, std::forward<HandleEvent>(handle_event), chunk_size);
    }

    /// bytes_to_events decodes a byte iterator.
    template <type event_stream_type, typename ByteIterator>
    inline std::vector<event<event_stream_type>> bytes_to_events(uint64_t t0, sepia::header header, ByteIterator begin, ByteIterator end) {
        auto stream_handle_byte = handle_byte<event_stream_type>(header.width, header.height);
        event<event_stream_type> stream_event = {t0};
        std::vector<event<event_stream_type>> events;
        for (; begin != end; ++begin) {
            if (stream_handle_byte(static_cast<uint8_t>(*begin), stream_event)) {
                events.push_back(stream_event);
            }
        }
        return events;
    }

    /// count_events calculates the number of events in a stream.
    template <type event_stream_type>
    inline std::size_t count_events(std::istream& event_stream, std::size_t chunk_size = 1 << 16) {
        std::size_t count = 0;
        read_events<event_stream_type>(
            event_stream, [&count](sepia::event<event_stream_type>) { ++count; }, chunk_size);
        return count;
    }
    template <type event_stream_type>
    inline std::size_t count_events(std::unique_ptr<std::istream> event_stream, std::size_t chunk_size = 1 << 16) {
        return count_events<event_stream_type>(*event_stream, chunk_size);
    }

    /// indexed_observable reads bytes from a stream and dispatches events.
    /// An index is built during construction to seek arbitrary parts of the file.
    template <type event_stream_type>
    class indexed_observable : public any_observable {
        public:
        indexed_observable(
            std::unique_ptr<std::istream> event_stream,
            uint64_t keyframe_duration,
            std::size_t chunk_size) :
            any_observable(), _event_stream(std::move(event_stream)), _handle_byte(0, 0) {
            const auto header = read_header(*_event_stream);
            if (header.event_stream_type != event_stream_type) {
                throw unsupported_event_type();
            }
            _event = event<event_stream_type>{};
            _handle_byte = handle_byte<event_stream_type>(header.width, header.height);
            auto position = static_cast<std::size_t>(static_cast<std::streamoff>(_event_stream->tellg()));
            std::size_t byte_index = 0;
            uint64_t next_threshold = 0;
            auto handle_event = [&]() {
                if (_keyframes.empty()) {
                    auto relative_keyframe = _handle_byte.relative_keyframe();
                    next_threshold = relative_keyframe.t + keyframe_duration;
                    relative_keyframe.offset = position + byte_index - relative_keyframe.offset;
                    _keyframes.push_back(relative_keyframe);
                } else if (_event.t >= next_threshold) {
                    auto relative_keyframe = _handle_byte.relative_keyframe();
                    relative_keyframe.offset = position + byte_index - relative_keyframe.offset;
                    const auto overflows = 1 + (_event.t - next_threshold) / keyframe_duration;
                    for (std::size_t index = 0; index < overflows; ++index) {
                        _keyframes.push_back(relative_keyframe);
                    }
                    next_threshold += keyframe_duration * overflows;
                }
            };
            _bytes.resize(chunk_size);
            for (;;) {
                _event_stream->read(reinterpret_cast<char*>(_bytes.data()), _bytes.size());
                byte_index = 0;
                if (_event_stream->eof()) {
                    for (auto byte_iterator = _bytes.begin();
                         byte_iterator
                         != std::next(
                             _bytes.begin(),
                             static_cast<std::iterator_traits<std::vector<uint8_t>::iterator>::difference_type>(
                                 _event_stream->gcount()));
                         ++byte_iterator) {
                        ++byte_index;
                        if (_handle_byte(*byte_iterator, _event)) {
                            handle_event();
                        }
                    }
                    break;
                } else {
                    for (auto byte : _bytes) {
                        ++byte_index;
                        if (_handle_byte(byte, _event)) {
                            handle_event();
                        }
                    }
                }
                position += _bytes.size();
            }
            _keyframes.push_back({position + byte_index, 0});
            _event_stream->clear();
            _event_stream->seekg(_keyframes.front().offset);
            _handle_byte.reset();
            _event.t = _keyframes.front().t;
        }

        indexed_observable(const indexed_observable&) = delete;
        indexed_observable(indexed_observable&&) = default;
        indexed_observable& operator=(const indexed_observable&) = delete;
        indexed_observable& operator=(indexed_observable&&) = default;
        virtual ~indexed_observable() {}

        /// keyframes returns the number of possible ckunks.
        /// The largest valid keyframe index is keyframes - 1.
        std::size_t keyframes() const {
            return _keyframes.size() - 1;
        }

        /// chunk retrieves the events in the range [keyframe_index, keyframe_index + 1[.
        const std::vector<sepia::event<event_stream_type>>& chunk(std::size_t keyframe_index) {
            if (keyframe_index >= _keyframes.size() - 1) {
                throw std::runtime_error(
                    std::string("the keyframe index must in the range [0, ") + std::to_string(_keyframes.size() - 2)
                    + "]");
            }
            const auto keyframe = _keyframes[keyframe_index];
            _event_stream->seekg(keyframe.offset);
            _handle_byte.reset();
            _event.t = keyframe.t;
            _bytes.resize(_keyframes[keyframe_index + 1].offset - keyframe.offset);
            _event_stream->read(reinterpret_cast<char*>(_bytes.data()), _bytes.size());
            _buffer.clear();
            _buffer.reserve(_bytes.size());
            for (auto byte : _bytes) {
                if (_handle_byte(byte, _event)) {
                    _buffer.push_back(_event);
                }
            }
            return _buffer;
        }

        protected:
        std::unique_ptr<std::istream> _event_stream;
        handle_byte<event_stream_type> _handle_byte;
        event<event_stream_type> _event;
        std::vector<uint8_t> _bytes;
        std::vector<sepia::event<event_stream_type>> _buffer;
        std::vector<keyframe> _keyframes;
    };

    /// make_indexed_observable creates a smart pointer to an indexed event stream observable.
    template <type event_stream_type>
    inline std::unique_ptr<indexed_observable<event_stream_type>> make_indexed_observable(
        std::unique_ptr<std::istream> event_stream,
        uint64_t keyframe_duration,
        std::size_t chunk_size = 1 << 16) {
        return sepia::make_unique<indexed_observable<event_stream_type>>(
            std::move(event_stream), keyframe_duration, chunk_size);
    }
}
