#!/bin/bash


counter=1
cmp="cmp_checkpoint.ckp"
file="my_checkpoint.ckp.old"
end=".ckp"
until [ $counter -gt 100 ]
do
	rm $file$counter$end
	counter=$((counter+1))

	rm $cmp$counter
done

rm my_checkpoint.ckp
rm my_checkpoint.dat
rm my_checkpoint.json
