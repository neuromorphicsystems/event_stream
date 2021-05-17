udp_receiver = udpport("datagram", "IPV4", "LocalPort", 12345, "EnablePortSharing", true, "Timeout", 3600);
while true
    packet = read(udp_receiver, 1, "uint8");
    [header, events] = event_stream_udp(packet.Data);
    header
end
