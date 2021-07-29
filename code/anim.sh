#!/bin/sh

year_list=("2020" "2021")
#month_list=("10" "11" "12" "01" "02" "03" "04")
month_list=("01" "02" "03" "04" "10" "11" "12")

for y in ${year_list[@]}; do

    for m in ${month_list[@]}; do


	# determine last day month
	#----------------------------
	if (test $m == 02) then
	   end=28
	elif (test $m == 11) then
	   end=30
	elif (test $m == 04) then
	   end=30
	else
	   end=31
	fi


	echo ${y}${m}
	# animation
	python animation_cryo2ice_mean.py -g ESA_BD_GDR -p radar_fb -g ATL10 -b b1,b2,b3 -p laser_fb -d${y}${m}01,${y}${m}${end} -f OctMar_ESA -o Anim_${y}${m}

    done

done
