#!/bin/bash
touch ip_connected.txt  
for ip in 192.168.0.{0..15}
  do   
    s=`ping -c 1 $ip | grep 'icmp'`
    check=`expr "$s" : '64 bytes'`
    if [ "$check" -ge 5 ]
    then 
      echo $ip
      echo $ip >> ip_connected.txt
    fi
  done



