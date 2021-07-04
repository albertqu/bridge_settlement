from SETTINGS import *
import socket, time, sys, select, traceback, Queue

#connects (one client only), and reads any data (one message anticipated), then sends out a response (one)
max_count = 30

def server_socket(send = "" ):

    recv = []
    print("RPICam: Establishing Server Socket...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((RPI_C_IP, RPI_C_Port))
    sock.listen(1)
    sock.setblocking(0)
    print("RPICam: Waiting for RPI_L to Connect")
    message_queues = {}
    inputs = [sock]
    outputs = []

    try:
        while inputs:
                readable, writable, errored = select.select(inputs, [], [], query_time*max_count)
                print readable
                print writable
                print errored
                for s in readable:
                    if s is sock: #the socket itself is ready to accept a connection
                        print("RPICam: Connected")
                        connection, server_address = s.accept()
                        connection.setblocking(0)
                        inputs.append(connection)
                        message_queues[connection] = Queue.Queue()
                    else: #the socket has something that needs to be read
                        print("RPICam: Reading Data")
                        msg = s.recv(1)
                        rsv_msg_L = []
                        if msg: #read in the message if the queue not empty
                            while True:
                                if marker in msg:
                                    rsv_msg_L.append(msg[:msg.find(marker)])
                                    break
                                rsv_msg_L.append(msg)
                                msg = s.recv(1)
                            message_queues[s].put(send)
                            print("RPICam: Data Read Successfully")
                            print("received: " + str(rsv_msg_L))
                            recv = rsv_msg_L
                            if s not in outputs:
                                outputs.append(s)
                                print("RPICam: Data to be Sent")
                            s.send(send)
                            #print(inputs)
                            #print(outputs)
                        else: #no data, client disconnected, close socket
                            if s in outputs:
                                outputs.remove(s)
                            inputs.remove(s)
                            s.close()
                            print("RPICam: Socket Closed")
                            del message_queues[s]
                #for s in writable:
                #    try: #retrieve output queue message and send it
                #        next_msg = message_queues[s].get_nowait()
                #        s.send(next_msg)
                #        print("RPICam: Data Sent Successfully")
                #    except Queue.Empty:
                #        outputs.remove(s)
                #        print("RPICam: All Data Sent")
                #for s in errored:
                #    inputs.remove(s)
                #    if s in outputs:
                #        outputs.remove(s)
                #    s.close()
                #    del message_queues[s]
    except KeyboardInterrupt:
        if sock:
            sock.close
    except:
        traceback.print_exc(file=sys.stdout)

    return recv










