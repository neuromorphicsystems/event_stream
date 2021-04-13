import event_stream
import numpy
import pathlib
import random
dirname = pathlib.Path(__file__).resolve().parent

with event_stream.Encoder(dirname / 'media' / 'random_dvs.es', 'dvs', 1280, 720) as encoder:
    events = []
    t = 0
    for index in range(0, 100):
        t += random.randint(0, 10)
        events.append((
            t,
            random.randint(0, 1279),
            random.randint(0, 719),
            bool(random.getrandbits(1)),
        ))
    events = numpy.array(events, dtype=event_stream.dvs_dtype)
    encoder.write(events)
    print(events)

with event_stream.Decoder(dirname / 'media' / 'random_dvs.es') as decoder:
    for packet in decoder:
        print(packet)
