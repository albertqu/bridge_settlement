from SETTINGS import *
import socket, time, sys, select, traceback, Queue

max_count = 40

def client_socket(send_msg = ''):

    rcv = []
    print("RPI_L: Establishing Client Socket...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(0)
    sock.settimeout(max_count * query_time)

    try:
        for i in range(0,max_count-1):
            try:
                print("RPI_L: Waiting for RPICam to Come On..." + str(i) + "/" + str(max_count-1))
                sock.connect((RPI_C_IP, RPI_C_Port))
                print("RPI_L: Link Established")
                break
            except:
                time.sleep(query_time)
    except KeyboardInterrupt:
        if sock:
            sock.close()
    except:
        traceback.print_exc(file=sys.stdout)

    try:
        try:
            print("RPI_L: Sending Data")
            sock.send(send_msg)
            print("RPI_L: Data Sent")
            count = 0
            while True:
                print("RPI_L: Reading Data...")
                if count < max_count:
                    try:
                        #print("trying...")
                        msg = sock.recv(1)
                        #print("msg: " + str(msg))
                        if marker in msg:
                            rcv.append(msg[:msg.find(marker)])
                            break
                        rcv.append(msg)
                        count = count + 1
                    except KeyboardInterrupt:
                        if sock:
                            sock.close()
                        break
                    except:
                        print("RPI_L: ERROR: Reading Interrupted")
                        break
            sock.close()
        except KeyboardInterrupt:
            if sock:
                sock.close()
        except:
            traceback.print_exc(file=sys.stdout)
            print("RPI_L: Connect Attempt Failed")
            if sock:
                sock.close()
                sock = -1
    except KeyboardInterrupt:
        sock.close()
    except:
        if i == max_count - 1:
            print("RPI_L: RPI_C Never Connected")
        traceback.print_exc(file=sys.stdout)

    return rcv


