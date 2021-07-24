#!/bin/bash

set -e

# read arguments
for ARG in "$@"; do
  KEY=$(echo $ARG | cut -f1 -d=)
  VAL=$(echo $ARG | cut -f2 -d=)
  case "$KEY" in
    --host) HOST=$VAL ;;
    --port) PORT=$VAL ;;
    --procs) PROCS=$VAL ;;
    --daemon) DAEMON=$VAL ;;
    *) echo "Unknown argument: $KEY"; exit 1 ;;
  esac
done

if [ -z $HOST ]; then echo "worker: host unset"; exit 1; fi
if [ -z $PORT ]; then echo "port unset"; exit 1; fi
if [ -z $PROCS ]; then echo "procs unset"; exit 1; fi
if [ -z $DAEMON ]; then echo "daemon unset"; exit 1; fi

# load environment
. start_env.sh

# start ncpu workers
echo "Starting workers to connect to server on $HOST:$PORT"
abc-redis-worker --host=$HOST --port=$PORT --runtime=24h \
  --processes=$PROCS --daemon=$DAEMON
