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
