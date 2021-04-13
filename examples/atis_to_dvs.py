import event_stream
import numpy.lib.recfunctions
import pathlib
dirname = pathlib.Path(__file__).resolve().parent

decoder = event_stream.Decoder(dirname / 'media' / 'atis.es')
encoder = event_stream.Encoder(dirname / 'media' / 'atis_filtered.es', 'dvs', decoder.width, decoder.height)

for packet in decoder:
    packet = packet[packet['exposure'] == False] # select only DVS events
    packet = packet[['t', 'x', 'y', 'polarity']] # drop the 'exposure' field
    packet = numpy.lib.recfunctions.repack_fields(packet) # repack the fields to remove the hole left by 'exposure'
    packet.dtype.names = event_stream.dvs_dtype.names # rename the fields ('t', 'x', 'y', 'on')
    encoder.write(packet)
