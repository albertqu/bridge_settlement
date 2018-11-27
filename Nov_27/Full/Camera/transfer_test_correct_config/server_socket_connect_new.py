from SETTINGS import *
import socket, time, sys, select, traceback

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
    inputs = [sock]
    connect = False
    try:
        while inputs:
                readable, writable, errored = select.select(inputs, [], [], query_time*max_count)
                #print readable
                #print writable
                #print errored
                if not readable:
                    break #case of no connection
                for s in readable:
                    if s is sock and connect == False: #the socket itself is ready to accept a connection
                        print("RPICam: Connected")
                        connection, server_address = s.accept()
                        connection.setblocking(0)
                        inputs.append(connection)
                        connect = True
                    else: #the socket has something that needs to be read
                        print("RPICam: Reading Data")
                        msg = s.recv(1)
                        rsv_msg_L = []
                        if msg: #rea d in the message if the queue not empty
                            while True:
                                if marker in msg:
                                    rsv_msg_L.append(msg[:msg.find(marker)])
                                    break
                                rsv_msg_L.append(msg)
                                msg = s.recv(1)
                            print("RPICam: Data Read Successfully")
                            print("received: " + str(rsv_msg_L))
                            recv = rsv_msg_L
                            s.send(send) #immediately send back message response
                            print("RPICam: Data Sent")
                        else: #no data, client disconnected, close socket
                            inputs.remove(s) #remove socket from inputs
                            s.close() #close socket
                            print("RPICam: Accept Socket Closed")
                            print(inputs)
                            if connect == True:
                                inputs.remove(sock)
                                sock.close()
                                print("RPICam: Listen Socket Closed")

    except KeyboardInterrupt:
        if sock:
            sock.close
    except:
        traceback.print_exc(file=sys.stdout)

    return recv










