import event_stream

decoder = event_stream.UdpDecoder(12345)

for chunk in decoder:
    print('{} events, ts = [{} µs, {} µs]'.format(len(chunk), chunk['t'][0], chunk['t'][-1]))
