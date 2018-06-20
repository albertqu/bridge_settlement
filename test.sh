#!/bin/bash  
for ip in 192.168.0.{0..255}
  do   
    k=`ping -c 1 $ip | grep 'icmp'`
    if [ "$k" > 5 ]
    then echo $ip
    fi
  done



