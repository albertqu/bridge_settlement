import socket


class SteadySocket:

    MAXMSGLEN = 128
    EOFChar = b'!'

    def __init__(self, sockt=None):
        if sockt is None:
            self.sock = socket.socket()
        else:
            self.sock = sockt

    def connect(self, host, port):
        self.sock.connect((host, port))

    def send(self, data, flags=0):
        data_sent = 0
        tot_len = len(data)
        while data_sent < tot_len:
            try:
                sent = self.sock.send(data[data_sent:])
                assert sent != 0
                data_sent += sent
            except:
                raise RuntimeError("Connection Lost!")

    def recv(self):
        data_recv = []
        bytes_recv = 0
        count = 0
        while bytes_recv < self.MAXMSGLEN:
            msg = self.sock.recv(self.MAXMSGLEN - bytes_recv)
            end = msg.find(self.EOFChar)
            if end != -1:
                data_recv.append(msg[:end])
                break
            data_recv.append(msg)
            bytes_recv += len(msg)
        print(count)
        return b''.join(data_recv)

    def close(self):
        self.sock.close()



IP = '192.168.3.4'
port = 8000

rpisock = SteadySocket()
connected = False
while True:
    if not connected:
        try:
            rpisock.connect(IP, port)
            connected = True
            msg = input("type a message\n")
            rpisock.send(bytes(msg, 'utf-8'))
            data = rpisock.recv()
            print(data)
        except:
            pass
    else:
        print("already connected")
        msg = input("type a message\n")
        rpisock.send(bytes(msg, 'utf-8'))
        data = rpisock.recv()
        print(data)
