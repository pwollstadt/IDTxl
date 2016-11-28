#!/bin/bash
# run speed test on PID estimators from Tartu and Sydney

cd ~/repos/IDTxl/dev/tartu_pid

for run in {1..20}
do
	python3 time_series_pw.py -s 'xor' 0.5 $run
	python3 time_series_pw.py -s 'and' 0.5 $run
done
