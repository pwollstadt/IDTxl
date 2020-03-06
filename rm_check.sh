#!/bin/bash


counter=1
cmp="cmp_checkpoint.ckp"
file="my_checkpoint.ckp.old"
end=".ckp"
until [ $counter -gt 100 ]
do
	rm $file$counter$end
	rm $cmp$counter
	counter=$((counter+1))
done

rm my_checkpoint.ckp
rm my_checkpoint.dat
rm my_checkpoint.json
rm $cmp
rm $file
rm resume_checkpoint.ckp
rm run_checkpoint.ckp
rm resume_checkpoint.ckp.old
rm run_checkpoint.ckp.old
rm resume_checkpoint.dat
rm resume_checkpoint.json
rm run_checkpoint.dat
rm run_checkpoint.json

