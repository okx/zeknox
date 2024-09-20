#!/bin/bash

SIZES="10 12 14 16 18 20"
REPS=3
TSTAMP=`date '+%Y-%m-%d-%H-%M-%S'`
FILE="benchmark_$TSTAMP.txt"

make testgpu.exe

for SIZE in $SIZES; do
	for REP in `seq 1 $REPS`; do
		./testgpu.exe $SIZE >> $FILE
	done
done
