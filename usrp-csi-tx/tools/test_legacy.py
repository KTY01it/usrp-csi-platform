import uhd
usrp = uhd.usrp.MultiUSRP("")
print("RX", usrp.get_rx_num_channels(), "TX", usrp.get_tx_num_channels())
