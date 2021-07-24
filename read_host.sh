#!/bin/bash

# read arguments
for ARG in "$@"; do
  KEY=$(echo $ARG | cut -f1 -d=)
  VAL=$(echo $ARG | cut -f2 -d=)
  case "$KEY" in
    --out) OUT=$VAL ;;  # number of nodes
    --port) PORT=$VAL ;;
    *) echo "Unknown argument"; exit 1 ;;
  esac
done

SERVER_IP_LONG=`cat $OUT/ip.txt | tr '\n' ' '`
HOST=$(host $SERVER_IP_LONG | awk '{ print $4 }')
echo "Server running on host $HOST"

abc-redis-manager info --host=$HOST --port=$PORT
