#!/bin/bash
touch ip_connected.txt  
for ip in 192.168.0.{0..255}
  do   
    k=`ping -c 1 $ip | grep 'icmp'`
    if [ "$k" > 5 ]
    then 
      echo $ip
      echo $ip >> ip_connected.txt
    fi
  done



