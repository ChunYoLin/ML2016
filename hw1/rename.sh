for i in 6_*; do
	new='six_'${i#*_}
	mv $i $new
done
