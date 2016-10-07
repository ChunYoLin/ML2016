FILES="weights/$2_*"
for f in $FILES
do
	echo "validate model $f"
	python src/model.py vd $1 $f
done
