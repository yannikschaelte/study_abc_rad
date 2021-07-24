#!/bin/bash

set -e

# read arguments
for ARG in "$@"; do
  KEY=$(echo $ARG | cut -f1 -d=)
  VAL=$(echo $ARG | cut -f2 -d=)
  case "$KEY" in
    --file) FILE=$VAL ;;
    --host) HOST=$VAL ;;
    --port) PORT=$VAL ;;
    --procs) PROCS=$VAL ;;
    --daemon) DAEMON=$VAL ;;
    *) echo "Unknown argument: $KEY"; exit 1 ;;
  esac
done

if [ -z $FILE ]; then echo "file unset"; exit 1; fi
if [ -z $HOST ]; then echo "host unset"; exit 1; fi
if [ -z $PORT ]; then echo "port unset"; exit 1; fi
if [ -z $PROCS ]; then echo "procs unset"; exit 1; fi
if [ -z $DAEMON ]; then echo "daemon unset"; exit 1; fi

# load environment
. start_env.sh

# start ncpu-1 workers
echo "Starting workers"
abc-redis-worker --host=$HOST --port=$PORT --runtime=24h \
  --processes=$((PROCS-1)) --daemon=$DAEMON &

sleep 1

# run main program
echo "Starting main program"
python $FILE --host=$HOST --port=$PORT
