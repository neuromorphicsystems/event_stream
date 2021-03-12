#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "sepia.hpp"
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

/// stream_to_array returns a structured array with the required length to accomodate the given stream.
template <sepia::type event_stream_type>
PyArrayObject* allocate_array(npy_intp size) {
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
            throw std::runtime_error("PyList_SetItem failed");
        }
    }
    PyArray_Descr* dtype;
    if (PyArray_DescrConverter(python_names_and_types, &dtype) == NPY_FAIL) {
        throw std::runtime_error("PyArray_DescrConverter failed");
    }
    return reinterpret_cast<PyArrayObject*>(
        PyArray_NewFromDescr(&PyArray_Type, dtype, 1, &size, nullptr, nullptr, 0, nullptr));
}

/// python_path_to_string converts a path-like object to a string.
std::string python_path_to_string(PyObject* path) {
    if (PyUnicode_Check(path)) {
        return reinterpret_cast<const char*>(PyUnicode_DATA(path));
    }
    {
        const auto characters = PyBytes_AS_STRING(path);
        if (characters) {
            return characters;
        }
    }
    auto string_or_bytes = PyObject_CallMethod(path, "__fspath__", nullptr);
    if (string_or_bytes) {
        if (PyUnicode_Check(string_or_bytes)) {
            return reinterpret_cast<const char*>(PyUnicode_DATA(path));
        }
        const auto characters = PyBytes_AS_STRING(string_or_bytes);
        if (characters) {
            return characters;
        }
    }
    throw std::runtime_error("path must be a string, bytes, or a path-like object");
}

/// decoder represents a Event Stream file.
struct decoder {
    PyObject_HEAD PyObject* type;
    PyObject* width;
    PyObject* height;
    std::unique_ptr<sepia::generic_observable> observable;
    std::vector<uint8_t> offsets;
    sepia::type cpp_type;
};
PyObject* decoder_iter(PyObject* self) {
    Py_INCREF(self);
    return self;
}
PyObject* decoder_iternext(PyObject* self) {
    auto current = reinterpret_cast<decoder*>(self);
    const auto& offsets = current->offsets;
    switch (current->cpp_type) {
        case sepia::type::generic: {
            const auto& buffer =
                static_cast<sepia::observable<sepia::type::generic>*>(current->observable.get())->next();
            if (buffer.empty()) {
                return nullptr;
            }
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
        case sepia::type::dvs: {
            const auto& buffer = static_cast<sepia::observable<sepia::type::dvs>*>(current->observable.get())->next();
            if (buffer.empty()) {
                return nullptr;
            }
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
        case sepia::type::atis: {
            const auto& buffer = static_cast<sepia::observable<sepia::type::atis>*>(current->observable.get())->next();
            if (buffer.empty()) {
                return nullptr;
            }
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
        case sepia::type::color: {
            const auto& buffer = static_cast<sepia::observable<sepia::type::color>*>(current->observable.get())->next();
            if (buffer.empty()) {
                return nullptr;
            }
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
        default:
            break;
    }
}
static PyMethodDef decoder_methods[] = {
    {nullptr, nullptr, 0, nullptr},
};
static PyMemberDef decoder_members[] = {
    {"type", T_OBJECT, offsetof(decoder, type), 0, "The Event Stream type"},
    {"width", T_OBJECT, offsetof(decoder, width), 0, "The sensor width in pixels"},
    {"height", T_OBJECT, offsetof(decoder, height), 0, "The Sensor height in pixels"},
    {nullptr, 0, 0, 0, nullptr},
};
static PyObject* decoder_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto current = reinterpret_cast<decoder*>(type->tp_alloc(type, 0));
    Py_INCREF(Py_None);
    current->type = Py_None;
    Py_INCREF(Py_None);
    current->width = Py_None;
    Py_INCREF(Py_None);
    current->height = Py_None;
    return reinterpret_cast<PyObject*>(current);
}
static int decoder_init(PyObject* self, PyObject* args, PyObject* kwds) {
    PyObject* path;
    if (!PyArg_ParseTuple(args, "O", &path)) {
        return -1;
    }
    auto current = reinterpret_cast<decoder*>(self);
    try {
        const auto filename = python_path_to_string(path);
        const auto header = sepia::read_header(sepia::filename_to_ifstream(filename));
        switch (header.event_stream_type) {
            case sepia::type::generic: {
                current->type = PyUnicode_FromString("generic");
                Py_INCREF(Py_None);
                current->width = Py_None;
                Py_INCREF(Py_None);
                current->height = Py_None;
                current->observable =
                    sepia::make_observable<sepia::type::generic>(sepia::filename_to_ifstream(filename));
                current->offsets = get_offsets<sepia::type::generic>();
                break;
            }
            case sepia::type::dvs: {
                current->type = PyUnicode_FromString("dvs");
                current->width = PyLong_FromLong(header.width);
                current->height = PyLong_FromLong(header.height);
                current->observable = sepia::make_observable<sepia::type::dvs>(sepia::filename_to_ifstream(filename));
                current->offsets = get_offsets<sepia::type::dvs>();
                break;
            }
            case sepia::type::atis: {
                current->type = PyUnicode_FromString("atis");
                current->width = PyLong_FromLong(header.width);
                current->height = PyLong_FromLong(header.height);
                current->observable = sepia::make_observable<sepia::type::atis>(sepia::filename_to_ifstream(filename));
                current->offsets = get_offsets<sepia::type::atis>();
                break;
            }
            case sepia::type::color: {
                current->type = PyUnicode_FromString("color");
                current->width = PyLong_FromLong(header.width);
                current->height = PyLong_FromLong(header.height);
                current->observable = sepia::make_observable<sepia::type::color>(sepia::filename_to_ifstream(filename));
                current->offsets = get_offsets<sepia::type::color>();
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

static PyMethodDef event_stream_methods[] = {{nullptr, nullptr, 0, nullptr}};
static struct PyModuleDef event_stream_definition =
    {PyModuleDef_HEAD_INIT, "event_stream", "event_stream reads Event Stream files", -1, event_stream_methods};
PyMODINIT_FUNC PyInit_event_stream() {
    auto module = PyModule_Create(&event_stream_definition);
    import_array();
    decoder_type.tp_name = "event_stream.Decoder";
    decoder_type.tp_basicsize = sizeof(decoder);
    decoder_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    decoder_type.tp_iter = decoder_iter;
    decoder_type.tp_iternext = decoder_iternext;
    decoder_type.tp_methods = decoder_methods;
    decoder_type.tp_members = decoder_members;
    decoder_type.tp_new = decoder_new;
    decoder_type.tp_init = decoder_init;
    PyType_Ready(&decoder_type);
    PyModule_AddObject(module, "Decoder", (PyObject*)&decoder_type);
    return module;
}
