import event_stream
import numpy.lib.recfunctions
import pathlib
dirname = pathlib.Path(__file__).resolve().parent

t_window = 2000

decoder = event_stream.Decoder(dirname / 'media' / 'dvs.es')
encoder = event_stream.Encoder(dirname / 'media' / 'dvs_filtered.es', decoder.type, decoder.width, decoder.height)

ts = numpy.zeros((decoder.width, decoder.height), dtype='<u8')
for packet in decoder:
    selection = numpy.zeros(len(packet), dtype='?')
    for index, (t, x, y, on) in enumerate(packet):
        ts[x, y] = t + t_window
        selection[index] = (
            (x > 0 and ts[x - 1, y] > t)
            or (y > 0 and ts[x, y - 1] > t)
            or (x < decoder.width - 1 and ts[x + 1, y] > t)
            or (y < decoder.height - 1 and ts[x, y + 1] > t))
    encoder.write(packet[selection])
