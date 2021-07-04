from Read_Accel import read_accel

while True:
    accel = raw_input('Press [y] to continue to acceL_test taking: ')
    if accel == 'y':
        try:
            print('Taking Accel Value...')
            Ax, Ay, Az = read_accel()
            print("Ax: " + str(Ax) + " Ay: " + str(Ay) + "Az: " + str(Az))
        except:
            print('Error Taking Accel Value...')
        finally:
            break
    else:
        print('Please try again')