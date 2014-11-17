for i in {1..18}
do
    if [ ${i} -ge 10 ];then
        idx=${i}
    else
        idx="0${i}"
    fi
    name=(`ls ${idx}/*pd*`)
    new_name="/home/omar/data/IBSR_nifti_stripped/IBSR_${idx}/IBSR_${idx}_ana_strip_pd.nii"
    cp ${name} ${new_name}
done
