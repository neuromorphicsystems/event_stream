#pragma once

#include <atomic>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Mswsock.h>
#include <WinSock2.h>
#include <iphlpapi.h>
#include <stdio.h>
#include <windows.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace udp {

#ifdef _WIN32
    using socket_file_descriptor_t = SOCKET;
#else
    using socket_file_descriptor_t = int32_t;
#endif

    /// check throws if the function returns an error code.
    inline void check(int32_t error, const std::string& message) {
#ifdef _WIN32
        if (error == SOCKET_ERROR) {
#else
        if (error < 0) {
#endif
            throw std::logic_error(message + " failed");
        }
    }

    /// check_socket throws if the socket is not valid, and returns the file descriptor otherwise.
    inline socket_file_descriptor_t check_socket(socket_file_descriptor_t socket_file_descriptor) {
#ifdef _WIN32
        if (socket_file_descriptor == INVALID_SOCKET) {
#else
        if (socket_file_descriptor < 0) {
#endif
            throw std::logic_error("the socket is not valid");
        }
        return socket_file_descriptor;
    }

    /// close_socket terminates the connection with a client.
    inline void close_socket(socket_file_descriptor_t socket_file_descriptor) {
#ifdef _WIN32
        if (shutdown(socket_file_descriptor, SD_BOTH) == 0) {
            ::closesocket(socket_file_descriptor);
        }
#else
        if (shutdown(socket_file_descriptor, SHUT_RDWR) == 0) {
            ::close(socket_file_descriptor);
        }
#endif
    }

    class receiver {
        public:
        receiver() = default;
        receiver(uint16_t port) : _owner(true) {
#ifdef _WIN32
            WSADATA wsa_data;
            if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
                throw std::runtime_error("WSAStartup failed");
            }
#endif
            _socket_file_descriptor = check_socket(socket(AF_INET, SOCK_DGRAM, 0));
#ifdef _WIN32
            GUID guid = WSAID_WSARECVMSG;
            DWORD bytes_read = 0;
            if (WSAIoctl(
                    _socket_file_descriptor,
                    SIO_GET_EXTENSION_FUNCTION_POINTER,
                    &guid,
                    sizeof(guid),
                    &_wsa_recv_msg,
                    sizeof(_wsa_recv_msg),
                    &bytes_read,
                    nullptr,
                    nullptr)
                != 0) {
                throw std::runtime_error("WSARecvMsg is not available");
            }
#endif
            {
                sockaddr_in address;
                address.sin_family = AF_INET;
                address.sin_addr.s_addr = INADDR_ANY;
                address.sin_port = htons(port);
                check(
                    bind(_socket_file_descriptor, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
                    "binding the socket");
            }
            _buffer.resize(1 << 16);
        }
        receiver(const receiver&) = delete;
        receiver(receiver&& other) :
            _owner(true), _socket_file_descriptor(other._socket_file_descriptor), _buffer(std::move(other._buffer)) {
#ifdef _WIN32
            _wsa_recv_msg = other._wsa_recv_msg;
#endif
            other._owner = false;
        }
        receiver& operator=(const receiver&) = delete;
        receiver& operator=(receiver&& other) {
            if (_owner) {
                close_socket(_socket_file_descriptor);
#ifdef _WIN32
                WSACleanup();
#endif
            }
            _owner = true;
            _socket_file_descriptor = other._socket_file_descriptor;
            _buffer = std::move(other._buffer);
#ifdef _WIN32
            _wsa_recv_msg = other._wsa_recv_msg;
#endif
            other._owner = false;
            return *this;
        }
        ~receiver() {
            if (_owner) {
                close_socket(_socket_file_descriptor);
#ifdef _WIN32
                WSACleanup();
#endif
            }
        }

        std::vector<uint8_t>& next() {
            _buffer.resize(1 << 16);
#ifdef _WIN32
            WSAMSG message = {};
            WSABUF location;
            location.buf = reinterpret_cast<CHAR*>(_buffer.data());
            location.len = static_cast<ULONG>(_buffer.size());
            message.lpBuffers = &location;
            message.dwBufferCount = 1;
            DWORD bytes_read;
            if (_wsa_recv_msg(_socket_file_descriptor, &message, &bytes_read, nullptr, nullptr) != 0) {
                throw std::runtime_error("socket error");
            }
#else
            msghdr message = {};
            iovec location;
            location.iov_base = _buffer.data();
            location.iov_len = _buffer.size();
            message.msg_iov = &location;
            message.msg_iovlen = 1;
            const auto bytes_read = recvmsg(_socket_file_descriptor, &message, 0);
            if (bytes_read < 0) {
                throw std::runtime_error("socket error");
            }
#endif
            _buffer.resize(bytes_read);
            return _buffer;
        }

        protected:
        bool _owner;
        socket_file_descriptor_t _socket_file_descriptor;
        std::vector<uint8_t> _buffer;
#ifdef _WIN32
        LPFN_WSARECVMSG _wsa_recv_msg;
#endif
    };

    class transmitter {
        public:
        transmitter() : _owner(true) {
            _socket_file_descriptor = check_socket(socket(AF_INET, SOCK_DGRAM, 0));
            int32_t option = 1;
            check(
                setsockopt(
                    _socket_file_descriptor,
                    SOL_SOCKET,
                    SO_REUSEADDR,
                    reinterpret_cast<const char*>(&option),
                    sizeof(option)),
                "enabling local address re-use");
        }
        transmitter(const transmitter&) = delete;
        transmitter(transmitter&& other) : _owner(true), _socket_file_descriptor(other._socket_file_descriptor) {
            other._owner = false;
        }
        transmitter& operator=(const transmitter&) = delete;
        transmitter& operator=(transmitter&& other) {
            if (_owner) {
                close_socket(_socket_file_descriptor);
#ifdef _WIN32
                WSACleanup();
#endif
            }
            _owner = true;
            _socket_file_descriptor = other._socket_file_descriptor;
            other._owner = false;
            return *this;
        }
        virtual ~transmitter() {
            close_socket(_socket_file_descriptor);
#ifdef _WIN32
            WSACleanup();
#endif
        }

        virtual void
        send(const std::string& receiver_address, uint16_t receiver_port, const char* data, std::size_t size) {
#ifdef _WIN32
            sockaddr_in address;
            InetPton(AF_INET, receiver_address.c_str(), &address);
            sendto(
                _socket_file_descriptor,
                data,
                static_cast<int32_t>(size),
                0,
                reinterpret_cast<sockaddr*>(&address),
                sizeof(address));
#else
            sockaddr_in address;
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = inet_addr(receiver_address.c_str());
            address.sin_port = htons(receiver_port);
            sendto(_socket_file_descriptor, data, size, 0, reinterpret_cast<sockaddr*>(&address), sizeof(address));
#endif
        }

        protected:
        bool _owner;
        socket_file_descriptor_t _socket_file_descriptor;
    };
}
