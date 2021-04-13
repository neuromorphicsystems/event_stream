import event_stream
import numpy.lib.recfunctions
import pathlib
dirname = pathlib.Path(__file__).resolve().parent

decoder = event_stream.Decoder(dirname / 'media' / 'dvs.es')
encoder = event_stream.Encoder(dirname / 'media' / 'dvs_flipped.es', decoder.type, decoder.width, decoder.height)

for packet in decoder:
    packet['y'] = decoder.height - 1 - packet['y']
    encoder.write(packet)
