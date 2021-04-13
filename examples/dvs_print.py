import event_stream
import pathlib
dirname = pathlib.Path(__file__).resolve().parent

with event_stream.Decoder(dirname / 'media' / 'dvs.es') as decoder:
    for packet in decoder:
        print(packet)
