# Event Stream

This repository contains an [Event Stream](http://github.com/neuromorphic-paris/event_stream) Python reader implemented in C++, as well as Matlab bindings.

1. [Python](#Python)
2. [Matlab](#Matlab)

## Python

```
pip3 install event_stream
```

### Documentation

The `event_stream` library provides three classes: `Decoder`, `IndexedDecoder` and `Encoder`:

-   `Decoder` reads constant-size byte buffers from an Event Stream file and returns variable-size event buffers
-   `IndexedDecoder` reads the entire file when created (without storing events in memory) to build an index, and can be used to fetch events at arbitrary timestamps
-   `UdpDecoder` reads event-stream encoded UDP packets (each packet must start with a uint64 little-endian absolute timestamp then contain an ES compliant stream)
-   `Encoder` writes event buffers to a file

Use `Decoder` if you want to process every event without delay. Use `IndexedDecoder` if you need to move back and forth while reading the file, for example if your are writing a file player with a clickable timeline.

The first argument to `Decoder`, `IndexedDecoder` and `Encoder` is a file name. It must be a [path-like object](https://docs.python.org/3/glossary.html#term-path-like-object). `IndexedDecoder` takes a second argument, the keyframe duration in µs. `Encoder` takes three other arguments, the evvent type and the sensor's width and height.

All three classes are contexts managers compatible with the `with` operator.

The detailed documentation for each class consists in a commented example (see below). There are more examples in the _examples_ directory. Run _examples/download.py_ first to download the media files used by the example scripts (12.8 MB).

#### Decoder

```python
import event_stream

# Decoder's only argument is an Event Stream file path
# decoder is an iterator with 3 additional properties: type, width and height
#     type is one of 'generic', 'dvs', 'atis' and 'color'
#     if type is 'generic', both width and height are None
#     otherwise, width and height represent the sensor size in pixels
decoder = event_stream.Decoder('/path/to/file.es')
if decoder.type == 'generic':
    print('generic events')
else:
    print(f'{decoder.type} events, {decoder.width} x {decoder.height} sensor')

# chunk is a numpy array whose dtype depends on the decoder type:
#     generic: [('t', '<u8'), ('bytes', 'object')]
#     dvs: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
#     atis: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('exposure', '?'), ('polarity', '?')]
#     color: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('r', '?'), ('g', '?'), ('b', '?')]
# chunk always contains at least one event
for chunk in decoder:
    print('{} events, ts = [{} µs, {} µs]'.format(len(chunk), chunk['t'][0], chunk['t'][-1]))
```

#### IndexedDecoder

```python
import event_stream

# IndexedDecoder's first argument is an Event Stream file path
#     its second argument is the duration of each keyframe in µs
#     the first keyframe starts with the first event
#     all the keyframes are offset accordingly
# decoder is an object with 3 properties: type, width and height
#     type is one of 'generic', 'dvs', 'atis' and 'color'
#     if type is 'generic', both width and height are None
#     otherwise, width and height represent the sensor size in pixels
# decoder has two methods: keyframes and chunk
decoder = event_stream.IndexedDecoder('/path/to/file.es', 40000)
if decoder.type == 'generic':
    print('generic events')
else:
    print(f'{decoder.type} events, {decoder.width} x {decoder.height} sensor')

# number of generated keyframes (one every 40000 µs here)
keyframes = decoder.keyframes()

for keyframe_index in range(0, keyframes):
    # keyframe_index must be in the range [0, keyframes[
    # the returned events have timestamps in the range
    #     [keyframe_index * T, (keyframe_index + 1) * T[
    #     where T is the second argument passed to IndexedDecoder
    # chunk is a numpy array whose dtype depends on the decoder type:
    #     generic: [('t', '<u8'), ('bytes', 'object')]
    #     dvs: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
    #     atis: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('exposure', '?'), ('polarity', '?')]
    #     color: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('r', '?'), ('g', '?'), ('b', '?')]
    # chunk may be empty
    chunk = decoder.chunk(keyframe_index)
    if len(chunk) > 0:
        print('{} / {}, {} events, ts = [{} µs, {} µs]'.format(
            keyframe_index + 1,
            keyframes,
            len(chunk),
            chunk['t'][0],
            chunk['t'][-1]))
    else:
        print('{} / {}, 0 events'.format(keyframe_index + 1, keyframes))
```

#### UdpDecoder

```python
import event_stream

# UdpDecoder's only argument is the port to which to bind
# decoder is an iterator with 3 additional properties: type, width and height
#     type is one of 'generic', 'dvs', 'atis' and 'color'
#     if type is 'generic', both width and height are None
#     otherwise, width and height represent the sensor size in pixels
# The additional properties are updated everytime a packet is received
decoder = event_stream.UdpDecoder(12345)

# chunk is a numpy array whose dtype depends on the packet type:
#     generic: [('t', '<u8'), ('bytes', 'object')]
#     dvs: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
#     atis: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('exposure', '?'), ('polarity', '?')]
#     color: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('r', '?'), ('g', '?'), ('b', '?')]

for chunk in decoder:
    print('{} events, ts = [{} µs, {} µs]'.format(len(chunk), chunk['t'][0], chunk['t'][-1]))
```

#### Encoder

```py
# Encoder's first argument is an Event Stream file path
#     its second argument is the event type, one of 'generic', 'dvs', 'atis' and 'color'
#     its third and fourth arguments are the sensor's width and height in pixels
#     The width and height are ignored if type is 'generic'
encoder = event_stream.Encoder('/path/to/file.es', 'dvs', 1280, 720)

# write adds an event buffer to the file
# the events must be sorted in order of increasing timestamp
# the chunk passed to write must be a structured numpy array whose dtype depends on the event type:
#     generic: [('t', '<u8'), ('bytes', 'object')]
#     dvs: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
#     atis: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('exposure', '?'), ('polarity', '?')]
#     color: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('r', '?'), ('g', '?'), ('b', '?')]
first_chunk = numpy.array([
    (0, 50, 100, True),
    (100, 1203, 641, False),
    (200, 73, 288, False),
    (300, 901, 99, True),
], dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')])
encoder.write(first_chunk)

# for convenience, event_stream provides dtype constants:
#     generic_dtype, dvs_dtype, atis_dtype and color_dtype
second_chunk = numpy.array([
    (400, 50, 100, True),
    (400, 1203, 641, False),
    (401, 73, 288, False),
    (401, 901, 99, True),
], dtype=event_stream.dvs_dtype)
encoder.write(second_chunk)
```

## Matlab

### Setup

After downloading this repository ([zip file](https://github.com/neuromorphicsystems/event_stream/archive/refs/heads/main.zip)), run the following commands in Matlab:

```js
cd /path/to/event_stream
cd matlab
mex event_stream_decode.cpp
mex event_stream_encode.cpp
mex event_stream_udp.cpp
```

The generated files (extension `.mexa64`, `.mexmaci64` or `.mexw64` depending on your operating system) can be placed in any directory. They contain the functions `event_stream_decode`, `event_stream_encode` and `event_stream_udp`. You can remove the rest of the repositrory from your machine if you want.

### Documentation

#### event_stream_decode

`event_stream_decode` reads events from a file.

```Matlab
[header, events] = event_stream_decode('/path/to/file.es');
```

```Matlab
header =

  struct with fields:

      type: 'dvs'
     width: 320
    height: 240

events =

  struct with fields:

     t: [473225×1 uint64]
     x: [473225×1 uint16]
     y: [473225×1 uint16]
    on: [473225×1 logical]
```

`header` is a struct with at least one field, `type`. `header.type` is either `'generic'`, `'dvs'`, `'atis'` or `'color'`. Unless `header.type` is `'generic'`, `header` has two extra fields, `width` and `height`. They encode the sensor size in pixels.

`events` is a struct whose fields are numerical arrays of equal length. Each array encodes one property of the events in the file (for example the timestamp `t`). The number of fields and their names depend on `header.type`:

-   `'generic'`:
    -   `t: [n×1 uint64]`
    -   `bytes: [n×1 string]`
-   `'dvs'`:
    -   `t: [n×1 uint64]`
    -   `x: [nx1 uint16]`
    -   `y: [nx1 uint16]`
    -   `on: [nx1 logical]`
-   `'atis'`:
    -   `t: [n×1 uint64]`
    -   `x: [nx1 uint16]`
    -   `y: [nx1 uint16]`
    -   `exposure: [nx1 logical]`
    -   `polarity: [nx1 logical]`
-   `'color'`:
    -   `t: [n×1 uint64]`
    -   `x: [nx1 uint16]`
    -   `y: [nx1 uint16]`
    -   `r: [nx1 uint8]`
    -   `g: [nx1 uint8]`
    -   `b: [nx1 uint8]`

#### event_stream_encode

`event_stream_encode` writes events to a file. The fields names and types must match those returned by `event_stream_decode`.

```Matlab
header = struct(...
    'type', 'dvs',...
    'width', uint16(320),...
    'height', uint16(240))

events = struct(...
    't', uint64([100; 200; 300]),...
    'x', uint16([303; 4; 42]),...
    'y', uint16([105; 201; 6]),...
    'on', logical([true; true; false]))

event_stream_encode('/path/to/file.es', header, events);
```

#### event_stream_udp

```Matlab
udp_receiver = udpport(
    "datagram",
    "IPV4",
    "LocalPort", 12345,
    "EnablePortSharing", true,
    "Timeout", 3600
);
while true
    packet = read(udp_receiver, 1, "uint8");
    [header, events] = event_stream_udp(packet.Data)
end
```

`header` and `events` have the same structure as the data returned by [event_stream_decode](#event_stream_decode).

## Contribute

To format the code, run:

```sh
clang-format -i sepia.hpp python/*.cpp matlab/*.cpp
```

## Publish

1. Bump the version number in _setup.py_.

2. Create a new release on GitHub.
