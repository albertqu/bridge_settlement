#!/bin/bash/

line=`ifconfig |grep broadcast`
echo $line
set -- $line
ip=$2
echo $ip
python manage.py runserver $ip:8000

