from sensors.models import ErrorStatus


def load_error_status():
    fl = r"static/Errors.txt"
    with open(fl, 'r') as f:
        for line in f.readlines():
            if line.count(":") >= 2:
                line = line.strip("\n")
                tg = line.find(":")
                print(line[:tg], line[tg + 2:])
                code, status = int(line[:tg]), line[tg+2:]
                if len(ErrorStatus.objects.filter(pk=code)) == 0:
                    print("Error status {}-{} created".format(code, status))
                    ErrorStatus.objects.create(code=code, status=status)


if __name__ == '__main__':
    load_error_status()