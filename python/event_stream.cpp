#include <Python.h>
#include <cstring>
#include <sstream>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../sepia.hpp"
#include "../udp.hpp"
#include <numpy/arrayobject.h>

/// description represents a named type with an offset.
struct description {
    std::string name;
    NPY_TYPES type;
};

/// descriptions returns the fields names, scalar types and offsets associated with an event type.
template <sepia::type event_stream_type>
std::vector<description> get_descriptions();
template <>
std::vector<description> get_descriptions<sepia::type::generic>() {
    return {{"t", NPY_UINT64}, {"bytes", NPY_OBJECT}};
}
template <>
std::vector<description> get_descriptions<sepia::type::dvs>() {
    return {{"t", NPY_UINT64}, {"x", NPY_UINT16}, {"y", NPY_UINT16}, {"on", NPY_BOOL}};
}
template <>
std::vector<description> get_descriptions<sepia::type::atis>() {
    return {{"t", NPY_UINT64}, {"x", NPY_UINT16}, {"y", NPY_UINT16}, {"exposure", NPY_BOOL}, {"polarity", NPY_BOOL}};
}
template <>
std::vector<description> get_descriptions<sepia::type::color>() {
    return {
        {"t", NPY_UINT64}, {"x", NPY_UINT16}, {"y", NPY_UINT16}, {"r", NPY_UINT8}, {"g", NPY_UINT8}, {"b", NPY_UINT8}};
}

/// offsets calculates the packed offsets from the description types.
template <sepia::type event_stream_type>
std::vector<uint8_t> get_offsets() {
    auto descriptions = get_descriptions<event_stream_type>();
    std::vector<uint8_t> offsets(descriptions.size(), 0);
    for (std::size_t index = 1; index < descriptions.size(); ++index) {
        switch (descriptions[index - 1].type) {
            case NPY_BOOL:
            case NPY_UINT8:
                offsets[index] = offsets[index - 1] + 1;
                break;
            case NPY_UINT16:
                offsets[index] = offsets[index - 1] + 2;
                break;
            case NPY_UINT64:
                offsets[index] = offsets[index - 1] + 8;
                break;
            default:
                throw std::runtime_error("unknown type for offset calculation");
        }
    }
    return offsets;
}

/// event_type_to_dtype returns a PyArray_Descr object.
template <sepia::type event_stream_type>
PyArray_Descr* event_type_to_dtype() {
    const auto descriptions = get_descriptions<event_stream_type>();
    auto python_names_and_types = PyList_New(static_cast<Py_ssize_t>(descriptions.size()));
    for (Py_ssize_t index = 0; index < static_cast<Py_ssize_t>(descriptions.size()); ++index) {
        if (PyList_SetItem(
                python_names_and_types,
                index,
                PyTuple_Pack(
                    2,
                    PyUnicode_FromString(descriptions[index].name.c_str()),
                    PyArray_TypeObjectFromType(descriptions[index].type)))
            < 0) {
            throw std::logic_error("PyList_SetItem failed");
        }
    }
    PyArray_Descr* dtype;
    if (PyArray_DescrConverter(python_names_and_types, &dtype) == NPY_FAIL) {
        throw std::logic_error("PyArray_DescrConverter failed");
    }
    return dtype;
}

/// allocate_array returns a structured array with the required length to accomodate the given stream.
template <sepia::type event_stream_type>
PyArrayObject* allocate_array(npy_intp size) {
    return reinterpret_cast<PyArrayObject*>(PyArray_NewFromDescr(
        &PyArray_Type, event_type_to_dtype<event_stream_type>(), 1, &size, nullptr, nullptr, 0, nullptr));
}

/// events_to_array converts a buffer of events to a numpy array.
template <sepia::type event_stream_type>
PyObject*
events_to_array(const std::vector<sepia::event<event_stream_type>>& buffer, const std::vector<uint8_t>& offsets);
template <>
PyObject* events_to_array(const std::vector<sepia::generic_event>& buffer, const std::vector<uint8_t>& offsets) {
    auto events = allocate_array<sepia::type::generic>(buffer.size());
    for (npy_intp index = 0; index < static_cast<npy_intp>(buffer.size()); ++index) {
        const auto generic_event = buffer[index];
        auto payload = reinterpret_cast<uint8_t*>(PyArray_GETPTR1(events, index));
        *reinterpret_cast<uint64_t*>(payload + offsets[0]) = generic_event.t;
        *reinterpret_cast<PyObject**>(payload + offsets[1]) = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(generic_event.bytes.data()), generic_event.bytes.size());
    }
    return reinterpret_cast<PyObject*>(events);
}
template <>
PyObject* events_to_array(const std::vector<sepia::dvs_event>& buffer, const std::vector<uint8_t>& offsets) {
    auto events = allocate_array<sepia::type::dvs>(buffer.size());
    for (npy_intp index = 0; index < static_cast<npy_intp>(buffer.size()); ++index) {
        const auto dvs_event = buffer[index];
        auto payload = reinterpret_cast<uint8_t*>(PyArray_GETPTR1(events, index));
        *reinterpret_cast<uint64_t*>(payload + offsets[0]) = dvs_event.t;
        *reinterpret_cast<uint16_t*>(payload + offsets[1]) = dvs_event.x;
        *reinterpret_cast<uint16_t*>(payload + offsets[2]) = dvs_event.y;
        *reinterpret_cast<bool*>(payload + offsets[3]) = dvs_event.on;
    }
    return reinterpret_cast<PyObject*>(events);
}
template <>
PyObject* events_to_array(const std::vector<sepia::atis_event>& buffer, const std::vector<uint8_t>& offsets) {
    auto events = allocate_array<sepia::type::atis>(buffer.size());
    for (npy_intp index = 0; index < static_cast<npy_intp>(buffer.size()); ++index) {
        const auto atis_event = buffer[index];
        auto payload = reinterpret_cast<uint8_t*>(PyArray_GETPTR1(events, index));
        *reinterpret_cast<uint64_t*>(payload + offsets[0]) = atis_event.t;
        *reinterpret_cast<uint16_t*>(payload + offsets[1]) = atis_event.x;
        *reinterpret_cast<uint16_t*>(payload + offsets[2]) = atis_event.y;
        *reinterpret_cast<bool*>(payload + offsets[3]) = atis_event.exposure;
        *reinterpret_cast<bool*>(payload + offsets[4]) = atis_event.polarity;
    }
    return reinterpret_cast<PyObject*>(events);
}
template <>
PyObject* events_to_array(const std::vector<sepia::color_event>& buffer, const std::vector<uint8_t>& offsets) {
    auto events = allocate_array<sepia::type::color>(buffer.size());
    for (npy_intp index = 0; index < static_cast<npy_intp>(buffer.size()); ++index) {
        const auto color_event = buffer[index];
        auto payload = reinterpret_cast<uint8_t*>(PyArray_GETPTR1(events, index));
        *reinterpret_cast<uint64_t*>(payload + offsets[0]) = color_event.t;
        *reinterpret_cast<uint16_t*>(payload + offsets[1]) = color_event.x;
        *reinterpret_cast<uint16_t*>(payload + offsets[2]) = color_event.y;
        *reinterpret_cast<uint8_t*>(payload + offsets[3]) = color_event.r;
        *reinterpret_cast<uint8_t*>(payload + offsets[4]) = color_event.g;
        *reinterpret_cast<uint8_t*>(payload + offsets[5]) = color_event.b;
    }
    return reinterpret_cast<PyObject*>(events);
}

/// python_path_to_string converts a path-like object to a string.
std::string python_path_to_string(PyObject* path) {
    if (PyUnicode_Check(path)) {
        return reinterpret_cast<const char*>(PyUnicode_DATA(path));
    }
    {
        const auto characters = PyBytes_AsString(path);
        if (characters) {
            return characters;
        } else {
            PyErr_Clear();
        }
    }
    auto string_or_bytes = PyObject_CallMethod(path, "__fspath__", nullptr);
    if (string_or_bytes) {
        if (PyUnicode_Check(string_or_bytes)) {
            return reinterpret_cast<const char*>(PyUnicode_DATA(string_or_bytes));
        }
        const auto characters = PyBytes_AsString(string_or_bytes);
        if (characters) {
            return characters;
        } else {
            PyErr_Clear();
        }
    }
    throw std::runtime_error("path must be a string, bytes, or a path-like object");
}

// int32_to_uint16 converts with checks an integer to a unsigned short.
uint16_t int32_to_uint16(int32_t value) {
    if (value < 0 || value >= (1 << 16)) {
        throw std::runtime_error("width and height must be in the range [0, 65535]");
    }
    return static_cast<uint16_t>(value);
}

// chunk_to_array extracts a Numpy array from a generic object.
template <sepia::type event_stream_type>
PyArrayObject* chunk_to_array(PyObject* chunk, const std::vector<uint8_t>& offsets) {
    if (!PyArray_Check(chunk)) {
        throw std::runtime_error("chunk must be a numpy array");
    }
    auto events = reinterpret_cast<PyArrayObject*>(chunk);
    if (PyArray_NDIM(events) != 1) {
        throw std::runtime_error("chunk's dimension must be 1");
    }
    const auto descriptions = get_descriptions<event_stream_type>();
    auto fields = PyArray_DESCR(events)->fields;
    if (!PyMapping_Check(fields)) {
        throw std::runtime_error("chunk must be a structured array");
    }
    for (Py_ssize_t index = 0; index < static_cast<Py_ssize_t>(descriptions.size()); ++index) {
        auto field = PyMapping_GetItemString(fields, descriptions[index].name.c_str());
        if (!field) {
            throw std::runtime_error(
                std::string("chunk must be a structured array with a '") + descriptions[index].name + "' field");
        }
        if (reinterpret_cast<PyArray_Descr*>(PyTuple_GetItem(field, 0))->type_num != descriptions[index].type) {
            Py_DECREF(field);
            throw std::runtime_error(
                std::string("the field '") + descriptions[index].name + " must have the type "
                + std::to_string(descriptions[index].type));
        }
        if (PyLong_AsLong(PyTuple_GetItem(field, 1)) != offsets[index]) {
            Py_DECREF(field);
            throw std::runtime_error(
                std::string("the field '") + descriptions[index].name + "' must have the offset "
                + std::to_string(offsets[index]));
        }
        Py_DECREF(field);
    }
    return events;
}

/// any_decoder is a common implementation for decoder, indexed_decoder and udp_decoder.
struct any_decoder {
    PyObject_HEAD PyObject* type;
    PyObject* width;
    PyObject* height;
    std::unique_ptr<sepia::any_observable> observable;
    sepia::type cpp_type;
    std::vector<uint8_t> generic_offsets;
    std::vector<uint8_t> dvs_offsets;
    std::vector<uint8_t> atis_offsets;
    std::vector<uint8_t> color_offsets;
    udp::receiver udp_receiver;
};
void any_decoder_dealloc(PyObject* self) {
    auto current = reinterpret_cast<any_decoder*>(self);
    Py_DECREF(current->type);
    Py_DECREF(current->width);
    Py_DECREF(current->height);
    Py_TYPE(self)->tp_free(self);
}
static PyObject* any_decoder_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto current = reinterpret_cast<any_decoder*>(type->tp_alloc(type, 0));
    Py_INCREF(Py_None);
    current->type = Py_None;
    Py_INCREF(Py_None);
    current->width = Py_None;
    Py_INCREF(Py_None);
    current->height = Py_None;
    return reinterpret_cast<PyObject*>(current);
}
static PyObject* any_decoder_enter(PyObject* self, PyObject*) {
    Py_INCREF(self);
    return self;
}
static PyObject* any_decoder_exit(PyObject* self, PyObject*) {
    auto current = reinterpret_cast<any_decoder*>(self);
    current->observable.reset();
    Py_RETURN_FALSE;
}
static PyMemberDef any_decoder_members[] = {
    {"type", T_OBJECT, offsetof(any_decoder, type), 0, "The Event Stream type"},
    {"width", T_OBJECT, offsetof(any_decoder, width), 0, "The sensor width in pixels"},
    {"height", T_OBJECT, offsetof(any_decoder, height), 0, "The Sensor height in pixels"},
    {nullptr, 0, 0, 0, nullptr},
};

/// decoder iterates over an Event Stream file.
static PyObject* decoder_iter(PyObject* self) {
    Py_INCREF(self);
    return self;
}
static PyObject* decoder_iternext(PyObject* self) {
    auto current = reinterpret_cast<any_decoder*>(self);
    try {
        if (!current->observable) {
            throw std::runtime_error("the file is closed");
        }
        switch (current->cpp_type) {
            case sepia::type::generic: {
                const auto& buffer =
                    static_cast<sepia::observable<sepia::type::generic>*>(current->observable.get())->next();
                if (buffer.empty()) {
                    return nullptr;
                }
                return events_to_array<sepia::type::generic>(buffer, current->generic_offsets);
            }
            case sepia::type::dvs: {
                const auto& buffer =
                    static_cast<sepia::observable<sepia::type::dvs>*>(current->observable.get())->next();
                if (buffer.empty()) {
                    return nullptr;
                }
                return events_to_array<sepia::type::dvs>(buffer, current->dvs_offsets);
            }
            case sepia::type::atis: {
                const auto& buffer =
                    static_cast<sepia::observable<sepia::type::atis>*>(current->observable.get())->next();
                if (buffer.empty()) {
                    return nullptr;
                }
                return events_to_array<sepia::type::atis>(buffer, current->atis_offsets);
            }
            case sepia::type::color: {
                const auto& buffer =
                    static_cast<sepia::observable<sepia::type::color>*>(current->observable.get())->next();
                if (buffer.empty()) {
                    return nullptr;
                }
                return events_to_array<sepia::type::color>(buffer, current->color_offsets);
            }
        }
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}
static PyMethodDef decoder_methods[] = {
    {"__enter__", any_decoder_enter, METH_NOARGS, nullptr},
    {"__exit__", any_decoder_exit, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
static int decoder_init(PyObject* self, PyObject* args, PyObject* kwds) {
    PyObject* path;
    if (!PyArg_ParseTuple(args, "O", &path)) {
        return -1;
    }
    auto current = reinterpret_cast<any_decoder*>(self);
    try {
        current->generic_offsets = get_offsets<sepia::type::generic>();
        current->dvs_offsets = get_offsets<sepia::type::dvs>();
        current->atis_offsets = get_offsets<sepia::type::atis>();
        current->color_offsets = get_offsets<sepia::type::color>();
        const auto filename = python_path_to_string(path);
        const auto header = sepia::read_header(sepia::filename_to_ifstream(filename));
        switch (header.event_stream_type) {
            case sepia::type::generic: {
                current->type = PyUnicode_FromString("generic");
                Py_DECREF(Py_None);
                current->observable =
                    sepia::make_observable<sepia::type::generic>(sepia::filename_to_ifstream(filename));
                break;
            }
            case sepia::type::dvs: {
                current->type = PyUnicode_FromString("dvs");
                Py_DECREF(Py_None);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(Py_None);
                current->height = PyLong_FromLong(header.height);
                Py_DECREF(Py_None);
                current->observable = sepia::make_observable<sepia::type::dvs>(sepia::filename_to_ifstream(filename));
                break;
            }
            case sepia::type::atis: {
                current->type = PyUnicode_FromString("atis");
                Py_DECREF(Py_None);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(Py_None);
                current->height = PyLong_FromLong(header.height);
                Py_DECREF(Py_None);
                current->observable = sepia::make_observable<sepia::type::atis>(sepia::filename_to_ifstream(filename));
                break;
            }
            case sepia::type::color: {
                current->type = PyUnicode_FromString("color");
                Py_DECREF(Py_None);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(Py_None);
                current->height = PyLong_FromLong(header.height);
                Py_DECREF(Py_None);
                current->observable = sepia::make_observable<sepia::type::color>(sepia::filename_to_ifstream(filename));
                break;
            }
            default:
                break;
        }
        current->cpp_type = header.event_stream_type;
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return -1;
    }
    return 0;
}
static PyTypeObject decoder_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

/// indexed_decoder can extract buffers from an Event Stream file at specific timestamps.
static PyObject* indexed_decoder_keyframes(PyObject* self, PyObject*) {
    auto current = reinterpret_cast<any_decoder*>(self);
    try {
        if (!current->observable) {
            throw std::runtime_error("the file is closed");
        }
        switch (current->cpp_type) {
            case sepia::type::generic: {
                return PyLong_FromSize_t(
                    static_cast<sepia::indexed_observable<sepia::type::generic>*>(current->observable.get())
                        ->keyframes());
            }
            case sepia::type::dvs: {
                return PyLong_FromSize_t(
                    static_cast<sepia::indexed_observable<sepia::type::dvs>*>(current->observable.get())->keyframes());
            }
            case sepia::type::atis: {
                return PyLong_FromSize_t(
                    static_cast<sepia::indexed_observable<sepia::type::atis>*>(current->observable.get())->keyframes());
            }
            case sepia::type::color: {
                return PyLong_FromSize_t(
                    static_cast<sepia::indexed_observable<sepia::type::color>*>(current->observable.get())
                        ->keyframes());
            }
        }
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}
static PyObject* indexed_decoder_chunk(PyObject* self, PyObject* args) {
    Py_ssize_t keyframe_index;
    if (!PyArg_ParseTuple(args, "n", &keyframe_index)) {
        return nullptr;
    }
    auto current = reinterpret_cast<any_decoder*>(self);
    try {
        if (!current->observable) {
            throw std::runtime_error("the file is closed");
        }
        switch (current->cpp_type) {
            case sepia::type::generic: {
                return events_to_array<sepia::type::generic>(
                    static_cast<sepia::indexed_observable<sepia::type::generic>*>(current->observable.get())
                        ->chunk(keyframe_index),
                    current->generic_offsets);
            }
            case sepia::type::dvs: {
                return events_to_array<sepia::type::dvs>(
                    static_cast<sepia::indexed_observable<sepia::type::dvs>*>(current->observable.get())
                        ->chunk(keyframe_index),
                    current->dvs_offsets);
            }
            case sepia::type::atis: {
                return events_to_array<sepia::type::atis>(
                    static_cast<sepia::indexed_observable<sepia::type::atis>*>(current->observable.get())
                        ->chunk(keyframe_index),
                    current->atis_offsets);
            }
            case sepia::type::color: {
                return events_to_array<sepia::type::color>(
                    static_cast<sepia::indexed_observable<sepia::type::color>*>(current->observable.get())
                        ->chunk(keyframe_index),
                    current->color_offsets);
            }
        }
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}
static PyMethodDef indexed_decoder_methods[] = {
    {"__enter__", any_decoder_enter, METH_NOARGS, nullptr},
    {"__exit__", any_decoder_exit, METH_VARARGS, nullptr},
    {"keyframes", indexed_decoder_keyframes, METH_NOARGS, nullptr},
    {"chunk", indexed_decoder_chunk, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
static int indexed_decoder_init(PyObject* self, PyObject* args, PyObject* kwds) {
    PyObject* path;
    Py_ssize_t keyframe_duration;
    if (!PyArg_ParseTuple(args, "On", &path, &keyframe_duration)) {
        return -1;
    }
    auto current = reinterpret_cast<any_decoder*>(self);
    try {
        current->generic_offsets = get_offsets<sepia::type::generic>();
        current->dvs_offsets = get_offsets<sepia::type::dvs>();
        current->atis_offsets = get_offsets<sepia::type::atis>();
        current->color_offsets = get_offsets<sepia::type::color>();
        const auto filename = python_path_to_string(path);
        const auto header = sepia::read_header(sepia::filename_to_ifstream(filename));
        switch (header.event_stream_type) {
            case sepia::type::generic: {
                current->type = PyUnicode_FromString("generic");
                Py_DECREF(Py_None);
                current->observable = sepia::make_indexed_observable<sepia::type::generic>(
                    sepia::filename_to_ifstream(filename), keyframe_duration);
                break;
            }
            case sepia::type::dvs: {
                current->type = PyUnicode_FromString("dvs");
                Py_DECREF(Py_None);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(Py_None);
                current->height = PyLong_FromLong(header.height);
                Py_DECREF(Py_None);
                current->observable = sepia::make_indexed_observable<sepia::type::dvs>(
                    sepia::filename_to_ifstream(filename), keyframe_duration);
                break;
            }
            case sepia::type::atis: {
                current->type = PyUnicode_FromString("atis");
                Py_DECREF(Py_None);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(Py_None);
                current->height = PyLong_FromLong(header.height);
                Py_DECREF(Py_None);
                current->observable = sepia::make_indexed_observable<sepia::type::atis>(
                    sepia::filename_to_ifstream(filename), keyframe_duration);
                break;
            }
            case sepia::type::color: {
                current->type = PyUnicode_FromString("color");
                Py_DECREF(Py_None);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(Py_None);
                current->height = PyLong_FromLong(header.height);
                Py_DECREF(Py_None);
                current->observable = sepia::make_indexed_observable<sepia::type::color>(
                    sepia::filename_to_ifstream(filename), keyframe_duration);
                break;
            }
        }
        current->cpp_type = header.event_stream_type;
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return -1;
    }
    return 0;
}
static PyTypeObject indexed_decoder_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

/// udp_decoder reads UDP packets.
static PyObject* udp_decoder_iter(PyObject* self) {
    Py_INCREF(self);
    return self;
}
static PyObject* udp_decoder_iternext(PyObject* self) {
    auto current = reinterpret_cast<any_decoder*>(self);
    try {
        const auto& bytes_buffer = current->udp_receiver.next();
        if (bytes_buffer.size() < 8) {
            return reinterpret_cast<PyObject*>(allocate_array<sepia::type::generic>(0));
        }
        const auto t0 = *reinterpret_cast<const uint64_t*>(bytes_buffer.data());
        std::stringstream header_stream(std::string(
            std::next(bytes_buffer.begin(), 8),
            bytes_buffer.size() < 8 + 20 ? bytes_buffer.end() : std::next(bytes_buffer.begin(), 8 + 20)));
        const auto header = sepia::read_header(header_stream);
        switch (header.event_stream_type) {
            case sepia::type::generic: {
                Py_DECREF(current->type);
                current->type = PyUnicode_FromString("generic");
                Py_DECREF(current->width);
                Py_INCREF(Py_None);
                current->width = Py_None;
                Py_DECREF(current->height);
                Py_INCREF(Py_None);
                current->height= Py_None;
                const auto buffer = sepia::bytes_to_events<sepia::type::generic>(
                    t0, header, std::next(bytes_buffer.begin(), 8 + 16), bytes_buffer.end());
                return events_to_array<sepia::type::generic>(buffer, current->generic_offsets);
            }
            case sepia::type::dvs: {
                Py_DECREF(current->type);
                current->type = PyUnicode_FromString("dvs");
                Py_DECREF(current->width);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(current->height);
                current->height = PyLong_FromLong(header.height);
                const auto buffer = sepia::bytes_to_events<sepia::type::dvs>(
                    t0, header, std::next(bytes_buffer.begin(), 8 + 20), bytes_buffer.end());
                return events_to_array<sepia::type::dvs>(buffer, current->dvs_offsets);
            }
            case sepia::type::atis: {
                Py_DECREF(current->type);
                current->type = PyUnicode_FromString("atis");
                Py_DECREF(current->width);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(current->height);
                current->height = PyLong_FromLong(header.height);
                const auto buffer = sepia::bytes_to_events<sepia::type::atis>(
                    t0, header, std::next(bytes_buffer.begin(), 8 + 20), bytes_buffer.end());
                return events_to_array<sepia::type::atis>(buffer, current->atis_offsets);
            }
            case sepia::type::color: {
                 Py_DECREF(current->type);
                current->type = PyUnicode_FromString("color");
                Py_DECREF(current->width);
                current->width = PyLong_FromLong(header.width);
                Py_DECREF(current->height);
                current->height = PyLong_FromLong(header.height);
                const auto buffer = sepia::bytes_to_events<sepia::type::color>(
                    t0, header, std::next(bytes_buffer.begin(), 8 + 20), bytes_buffer.end());
                return events_to_array<sepia::type::color>(buffer, current->color_offsets);
            }
        }
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}
static PyMethodDef udp_decoder_methods[] = {
    {"__enter__", any_decoder_enter, METH_NOARGS, nullptr},
    {"__exit__", any_decoder_exit, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
static int udp_decoder_init(PyObject* self, PyObject* args, PyObject* kwds) {
    uint16_t port;
    if (!PyArg_ParseTuple(args, "H", &port)) {
        return -1;
    }
    auto current = reinterpret_cast<any_decoder*>(self);
    try {
        current->generic_offsets = get_offsets<sepia::type::generic>();
        current->dvs_offsets = get_offsets<sepia::type::dvs>();
        current->atis_offsets = get_offsets<sepia::type::atis>();
        current->color_offsets = get_offsets<sepia::type::color>();
        current->udp_receiver = udp::receiver(port);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return -1;
    }
    return 0;
}
static PyTypeObject udp_decoder_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

/// encoder writes to an Event Stream file.
struct encoder {
    PyObject_HEAD std::unique_ptr<sepia::any_write> write;
    std::vector<uint8_t> offsets;
    sepia::type cpp_type;
};
void encoder_dealloc(PyObject* self) {
    Py_TYPE(self)->tp_free(self);
}
static PyObject* encoder_new(PyTypeObject* type, PyObject*, PyObject*) {
    return type->tp_alloc(type, 0);
}
static PyMemberDef encoder_members[] = {
    {nullptr, 0, 0, 0, nullptr},
};
static PyObject* encoder_enter(PyObject* self, PyObject*) {
    Py_INCREF(self);
    return self;
}
static PyObject* encoder_exit(PyObject* self, PyObject*) {
    auto current = reinterpret_cast<encoder*>(self);
    current->write.reset();
    Py_RETURN_FALSE;
}
static PyObject* encoder_write(PyObject* self, PyObject* args) {
    PyObject* chunk;
    if (!PyArg_ParseTuple(args, "O", &chunk)) {
        return nullptr;
    }
    auto current = reinterpret_cast<encoder*>(self);
    try {
        if (!current->write) {
            throw std::runtime_error("the file is closed");
        }
        switch (current->cpp_type) {
            case sepia::type::generic: {
                auto events = chunk_to_array<sepia::type::generic>(chunk, current->offsets);
                auto& write = *static_cast<sepia::write<sepia::type::generic>*>(current->write.get());
                for (npy_intp index = 0; index < PyArray_SIZE(events); ++index) {
                    auto payload = reinterpret_cast<uint8_t*>(PyArray_GETPTR1(events, index));
                    auto bytes = *reinterpret_cast<PyObject**>(payload + current->offsets[1]);
                    if (!PyBytes_Check(bytes)) {
                        throw std::runtime_error("bytes must be a byte string");
                    }
                    std::string bytes_as_string(PyBytes_AsString(bytes));
                    write(
                        {*reinterpret_cast<uint64_t*>(payload + current->offsets[0]),
                         std::vector<uint8_t>(bytes_as_string.begin(), bytes_as_string.end())});
                }
                Py_RETURN_NONE;
            }
            case sepia::type::dvs: {
                auto events = chunk_to_array<sepia::type::dvs>(chunk, current->offsets);
                auto& write = *static_cast<sepia::write<sepia::type::dvs>*>(current->write.get());
                for (npy_intp index = 0; index < PyArray_SIZE(events); ++index) {
                    auto payload = reinterpret_cast<uint8_t*>(PyArray_GETPTR1(events, index));
                    write(
                        {*reinterpret_cast<uint64_t*>(payload + current->offsets[0]),
                         *reinterpret_cast<uint16_t*>(payload + current->offsets[1]),
                         *reinterpret_cast<uint16_t*>(payload + current->offsets[2]),
                         *reinterpret_cast<bool*>(payload + current->offsets[3])});
                }
                Py_RETURN_NONE;
            }
            case sepia::type::atis: {
                auto events = chunk_to_array<sepia::type::atis>(chunk, current->offsets);
                auto& write = *static_cast<sepia::write<sepia::type::atis>*>(current->write.get());
                for (npy_intp index = 0; index < PyArray_SIZE(events); ++index) {
                    auto payload = reinterpret_cast<uint8_t*>(PyArray_GETPTR1(events, index));
                    write(
                        {*reinterpret_cast<uint64_t*>(payload + current->offsets[0]),
                         *reinterpret_cast<uint16_t*>(payload + current->offsets[1]),
                         *reinterpret_cast<uint16_t*>(payload + current->offsets[2]),
                         *reinterpret_cast<bool*>(payload + current->offsets[3]),
                         *reinterpret_cast<bool*>(payload + current->offsets[4])});
                }
                Py_RETURN_NONE;
            }
            case sepia::type::color: {
                auto events = chunk_to_array<sepia::type::color>(chunk, current->offsets);
                auto& write = *static_cast<sepia::write<sepia::type::color>*>(current->write.get());
                for (npy_intp index = 0; index < PyArray_SIZE(events); ++index) {
                    auto payload = reinterpret_cast<uint8_t*>(PyArray_GETPTR1(events, index));
                    write(
                        {*reinterpret_cast<uint64_t*>(payload + current->offsets[0]),
                         *reinterpret_cast<uint16_t*>(payload + current->offsets[1]),
                         *reinterpret_cast<uint16_t*>(payload + current->offsets[2]),
                         *reinterpret_cast<uint8_t*>(payload + current->offsets[3]),
                         *reinterpret_cast<uint8_t*>(payload + current->offsets[4]),
                         *reinterpret_cast<uint8_t*>(payload + current->offsets[5])});
                }
                Py_RETURN_NONE;
            }
        }
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}
static PyMethodDef encoder_methods[] = {
    {"__enter__", encoder_enter, METH_NOARGS, nullptr},
    {"__exit__", encoder_exit, METH_VARARGS, nullptr},
    {"write", encoder_write, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
static int encoder_init(PyObject* self, PyObject* args, PyObject* kwds) {
    PyObject* path;
    const char* type;
    int32_t width;
    int32_t height;
    if (!PyArg_ParseTuple(args, "Osii", &path, &type, &width, &height)) {
        return -1;
    }
    auto current = reinterpret_cast<encoder*>(self);
    try {
        const auto filename = python_path_to_string(path);
        if (std::strcmp("generic", type) == 0) {
            current->write =
                sepia::make_unique<sepia::write<sepia::type::generic>>(sepia::filename_to_ofstream(filename));
            current->offsets = get_offsets<sepia::type::generic>();
            current->cpp_type = sepia::type::generic;
        } else if (std::strcmp("dvs", type) == 0) {
            current->write = sepia::make_unique<sepia::write<sepia::type::dvs>>(
                sepia::filename_to_ofstream(filename), int32_to_uint16(width), int32_to_uint16(height));
            current->offsets = get_offsets<sepia::type::dvs>();
            current->cpp_type = sepia::type::dvs;
        } else if (std::strcmp("atis", type) == 0) {
            current->write = sepia::make_unique<sepia::write<sepia::type::atis>>(
                sepia::filename_to_ofstream(filename), int32_to_uint16(width), int32_to_uint16(height));
            current->offsets = get_offsets<sepia::type::atis>();
            current->cpp_type = sepia::type::atis;
        } else if (std::strcmp("color", type) == 0) {
            current->write = sepia::make_unique<sepia::write<sepia::type::color>>(
                sepia::filename_to_ofstream(filename), int32_to_uint16(width), int32_to_uint16(height));
            current->offsets = get_offsets<sepia::type::color>();
            current->cpp_type = sepia::type::color;
        } else {
            throw std::runtime_error("type must be 'generic', 'dvs', 'atis' or 'color'");
        }
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        return -1;
    }
    return 0;
}
static PyTypeObject encoder_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

static PyMethodDef event_stream_methods[] = {{nullptr, nullptr, 0, nullptr}};
static struct PyModuleDef event_stream_definition =
    {PyModuleDef_HEAD_INIT, "event_stream", "event_stream reads Event Stream files", -1, event_stream_methods};
PyMODINIT_FUNC PyInit_event_stream() {
    auto module = PyModule_Create(&event_stream_definition);
    import_array();
    PyModule_AddObject(module, "generic_dtype", (PyObject*)event_type_to_dtype<sepia::type::generic>());
    PyModule_AddObject(module, "dvs_dtype", (PyObject*)event_type_to_dtype<sepia::type::dvs>());
    PyModule_AddObject(module, "atis_dtype", (PyObject*)event_type_to_dtype<sepia::type::atis>());
    PyModule_AddObject(module, "color_dtype", (PyObject*)event_type_to_dtype<sepia::type::color>());
    decoder_type.tp_name = "event_stream.Decoder";
    decoder_type.tp_basicsize = sizeof(any_decoder);
    decoder_type.tp_dealloc = any_decoder_dealloc;
    decoder_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    decoder_type.tp_iter = decoder_iter;
    decoder_type.tp_iternext = decoder_iternext;
    decoder_type.tp_methods = decoder_methods;
    decoder_type.tp_members = any_decoder_members;
    decoder_type.tp_new = any_decoder_new;
    decoder_type.tp_init = decoder_init;
    PyType_Ready(&decoder_type);
    PyModule_AddObject(module, "Decoder", (PyObject*)&decoder_type);
    indexed_decoder_type.tp_name = "event_stream.IndexedDecoder";
    indexed_decoder_type.tp_basicsize = sizeof(any_decoder);
    indexed_decoder_type.tp_dealloc = any_decoder_dealloc;
    indexed_decoder_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    indexed_decoder_type.tp_methods = indexed_decoder_methods;
    indexed_decoder_type.tp_members = any_decoder_members;
    indexed_decoder_type.tp_new = any_decoder_new;
    indexed_decoder_type.tp_init = indexed_decoder_init;
    PyType_Ready(&indexed_decoder_type);
    PyModule_AddObject(module, "IndexedDecoder", (PyObject*)&indexed_decoder_type);
    udp_decoder_type.tp_name = "event_stream.UdpDecoder";
    udp_decoder_type.tp_basicsize = sizeof(any_decoder);
    udp_decoder_type.tp_dealloc = any_decoder_dealloc;
    udp_decoder_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    udp_decoder_type.tp_iter = udp_decoder_iter;
    udp_decoder_type.tp_iternext = udp_decoder_iternext;
    udp_decoder_type.tp_methods = udp_decoder_methods;
    udp_decoder_type.tp_members = any_decoder_members;
    udp_decoder_type.tp_new = any_decoder_new;
    udp_decoder_type.tp_init = udp_decoder_init;
    PyType_Ready(&udp_decoder_type);
    PyModule_AddObject(module, "UdpDecoder", (PyObject*)&udp_decoder_type);
    encoder_type.tp_name = "event_stream.Encoder";
    encoder_type.tp_basicsize = sizeof(encoder);
    encoder_type.tp_dealloc = encoder_dealloc;
    encoder_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    encoder_type.tp_methods = encoder_methods;
    encoder_type.tp_members = encoder_members;
    encoder_type.tp_new = encoder_new;
    encoder_type.tp_init = encoder_init;
    PyType_Ready(&encoder_type);
    PyModule_AddObject(module, "Encoder", (PyObject*)&encoder_type);
    return module;
}
