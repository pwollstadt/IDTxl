#!/bin/bash

checkpoint=$1
index=1

end=".ckp"
DIR="/home/janosch_local/IDTxl/"
inotifywait -m -r -e create "$DIR" | while read f

do
    if [  -f $DIR$checkpoint ]; then
      mv "$DIR$checkpoint" "$DIR$checkpoint$index$end"
      index=$((index+1))
    fi
done
