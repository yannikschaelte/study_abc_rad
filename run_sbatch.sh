#!/bin/bash
#SBATCH --account=fitmulticell
#SBATCH --partition=batch
#SBATCH --time=24:00:00

set -e

CPUS_PER_TASK=48

# read arguments
for ARG in "$@"; do
  KEY=$(echo $ARG | cut -f1 -d=)
  VAL=$(echo $ARG | cut -f2 -d=)
  case "$KEY" in
    --account) ACCOUNT=$VAL ;;
    --nodes) NODES=$VAL ;;
    --out) OUT=$VAL ;;
    --file) FILE=$VAL ;;
    --port) PORT=$VAL ;;
    --daemon) DAEMON=$VAL ;;
    *) echo "Unknown argument: $KEY"; exit 1 ;;
  esac
done

if [ -z $ACCOUNT ]; then echo "account unset"; exit 1; fi
if [ -z $NODES ]; then echo "nodes unset"; exit 1; fi
if [ -z $OUT ]; then echo "out unset"; exit 1; fi
if [ -z $FILE ]; then echo "file unset"; exit 1; fi
if [ -z $PORT ]; then echo "port unset"; exit 1; fi
if [ -z $DAEMON ]; then echo "daemon unset"; exit 1; fi

sleep 5

# load environment
. start_env.sh

# start server
echo "Starting node with server"
srun \
  --exclusive \
  --nodes=1 --ntasks=1 --cpus-per-task=$CPUS_PER_TASK \
  --output=$OUT/out_slurm_server.txt --error=$OUT/err_slurm_server.txt \
  run_server.sh \
  --out=$OUT \
  --port=$PORT --daemon=$DAEMON --file=$FILE --procs=$CPUS_PER_TASK &

# give server time to start
sleep 10

# retrieve server id
HOSTNAME=`cat $OUT/ip.txt | tr '\n' ' '`
HOST=$(host $HOSTNAME | awk '{ print $4 }')
echo "Server running on IP $HOST"

# start workers
WORKERS=$((NODES-2))
echo "Starting $WORKERS nodes with workers"
for IW in $(seq $WORKERS); do
  srun \
    --exclusive \
    --nodes=1 --ntasks=1 --cpus-per-task=$CPUS_PER_TASK \
    --output=$OUT/out_slurm_worker_$IW.txt --error=$OUT/err_slurm_worker_$IW.txt \
    run_worker.sh \
    --host=$HOST --port=$PORT --daemon=$DAEMON --procs=$CPUS_PER_TASK &
  sleep 1
done

# start main program
echo "Starting node with main program"
srun \
  --exclusive \
  --nodes=1 --ntasks=1 --cpus-per-task=$CPUS_PER_TASK \
  --output=$OUT/out_slurm_main.txt --error=$OUT/err_slurm_main.txt \
  run_main.sh \
  --file=$FILE --host=$HOST --port=$PORT --daemon=$DAEMON --procs=$CPUS_PER_TASK
