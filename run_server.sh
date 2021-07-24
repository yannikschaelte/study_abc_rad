#!/bin/bash

set -e

# read arguments
for ARG in "$@"; do
  KEY=$(echo $ARG | cut -f1 -d=)
  VAL=$(echo $ARG | cut -f2 -d=)
  case "$KEY" in
    --out) OUT=$VAL ;;
    --file) FILE=$VAL ;;
    --port) PORT=$VAL ;;
    --procs) PROCS=$VAL ;;
    --daemon) DAEMON=$VAL ;;
    *) echo "Unknown argument: $KEY"; exit 1 ;;
  esac
done

if [ -z $OUT ]; then echo "out unset"; exit 1; fi
if [ -z $FILE ]; then echo "file unset"; exit 1; fi
if [ -z $PORT ]; then echo "port unset"; exit 1; fi
if [ -z $PROCS ]; then echo "procs unset"; exit 1; fi
if [ -z $DAEMON ]; then echo "daemon unset"; exit 1; fi

sleep 5

# load environment
. start_env.sh

hostname > $OUT/ip.txt
echo "Hostname is $(hostname)"

echo "Starting redis server"
redis-server --protected-mode no --port $PORT &

# give server time to start (TODO check if it is up)
sleep 5

# start ncpu-1 workers
echo "Starting workers"
abc-redis-worker --host=localhost --port=$PORT --runtime=24h \
  --processes=$((PROCS-1)) --daemon=$DAEMON
