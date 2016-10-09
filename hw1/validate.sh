FILES="weights/*"
for f in $FILES
do
	file=${f#weights/}
	model=${file%_Adagrad_*}
	echo ---------------------------------------------------
	echo "validate model $file"
	python src/model.py vd 'cfg/'${model}'.json' $f 
done
