# Event Stream

Event Stream is a fast and buffered [Event Stream](http://github.com/neuromorphic-paris/event_stream) python reader, with a C++ underlying implementation.

Run `pip install event_stream` to install it.

# Documentation

The `event_stream` library provides two classes: `Decoder` and `IndexedDecoder`:
- `Decoder` reads constant-size byte buffers from an Event Stream file and returns variable-size event buffers
- `IndexedDecoder` reads the entire file when created (without storing events in memory) to build an index that can be used afterwards to fetch events at arbitrary timestamps

Use `Decoder` if you want to process every event in order without delay. Use `IndexedDecoder` if you need to move back and forth while reading the file, for example in a file player with a cliackable timeline.

The first argument to `Decoder` and `IndexedDecoder` is a file name. It must be a [path-like object](https://docs.python.org/3/glossary.html#term-path-like-object). `IndexedDecoder` takes a second argument, the keyframe duration in µs.

Here's a `Decoder` example:
```python
import event_stream

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
for chunk in decoder:
    print('{} events, ts = [{} µs, {} µs]'.format(len(chunk), chunk['t'][0], chunk['t'][-1]))
```

Here's an `IndexedDecoder` example:

```python
import event_stream

# decoder is an object with 3 properties: type, width and height and two methods: keyframes and chunk
#     type is one of 'generic', 'dvs', 'atis' and 'color'
#     if type is 'generic', both width and height are None
#     otherwise, width and height represent the sensor size in pixels
decoder = event_stream.IndexedDecoder('/path/to/file.es', 40000)
if decoder.type == 'generic':
    print('generic events')
else:
    print(f'{decoder.type} events, {decoder.width} x {decoder.height} sensor')

keyframes = decoder.keyframes()
# number of generated keyframes (one every 40000 µs here)

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

# Contribute

To format the code, run:
```sh
clang-format -i source/sepia.hpp source/event_stream.cpp
```

# Publish

The version number can be changed in *setup.py*.

```sh
rm -rf dist
python3 setup.py sdist
twine upload dist/*
```
