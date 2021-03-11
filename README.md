# Event Stream

Event Stream is a fast and buffered [Event Stream](http://github.com/neuromorphic-paris/event_stream) python reader, with a C++ underlying implementation.

Run `pip install event_stream` to install it.

# Documentation

The `event_stream` library provides a single class: `Decoder`. A decoder object is created by passing a file name to `Decoder`. The file name must be a [path-like object](https://docs.python.org/3/glossary.html#term-path-like-object).

Here's a short example:
```python
import event_stream

decoder = event_stream.Decoder('/path/to/file.es')
"""
decoder is a packet iterator with 3 additional properties: type, width and height
type is one of 'generic', 'dvs', 'atis' and 'color'
if type is 'generic', both width and height are None
otherwise, width and height represent the sensor size in pixels
"""
if decoder.type == 'generic':
    print('generic events')
else:
    print(f'{decoder.type} events, {decoder.width} x {decoder.height} sensor')

for packet in decoder:
    """
    packet is a numpy array whose dtype depends on the decoder type:
        generic: [('t', '<u8'), ('bytes', 'object')]
        dvs: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
        atis: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('exposure', '?'), ('polarity', '?')]
        color: [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('r', '?'), ('g', '?'), ('b', '?')]
    """
    print('{} events, ts = [{} µs, {} µs]'.format(len(packet), packet['t'][0], packet['t'][-1]))
```

# Contribute

To format the code, run:
```sh
clang-format -i source/sepia.h source/event_stream.cpp
```

# Publish

The version number can be changed in *setup.py*.

```sh
rm -rf dist
python3 setup.py sdist
twine upload dist/*
```
