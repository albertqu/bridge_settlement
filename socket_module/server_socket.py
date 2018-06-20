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
        if data.find(self.EOFChar == -1):
            data += self.EOFChar
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


def main():
    ssock = socket.socket()
    IP = "192.168.3.4"
    port = 8000
    ssock.bind((IP, port))
    print("Now connected to " + IP + ":" + str(port))
    ssock.listen(1)

    conn = None
    addr = None
    sconn = None
    while True:
        if conn is None:
            conn, addr = ssock.accept()
            sconn = SteadySocket(conn)
            print(addr)
        else:
            data = sconn.recv()
            if data:
                print(data)
            if data == b"hi":
                sconn.send(b"how are you!")
            elif data == b'c':
                sconn.close()
                break
            else:
                sconn.send(b"It's good huh?!")
            inp = input("value?\n")
            if inp == 'e':
                sconn.close()
                break


if __name__ == "__main__":
    main()
