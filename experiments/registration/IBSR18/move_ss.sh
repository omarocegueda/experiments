for i in {1..18}
do
    if [ ${i} -ge 10 ];then
        idx=${i}
    else
        idx="0${i}"
    fi
    name=(`ls ${idx}/ss*t2*`)
    new_name="/home/omar/data/IBSR_nifti_stripped/IBSR_${idx}/IBSR_${idx}_ana_strip_t2_ds.nii.gz"
    echo mv ${name} ${new_name}
    name=(`ls ${idx}/ss*pd*`)
    new_name="/home/omar/data/IBSR_nifti_stripped/IBSR_${idx}/IBSR_${idx}_ana_strip_pd_ds.nii.gz"
    echo mv ${name} ${new_name}
    name=(`ls ${idx}/ss*t1*`)
    new_name="/home/omar/data/IBSR_nifti_stripped/IBSR_${idx}/IBSR_${idx}_ana_strip_t1_ds.nii.gz"
    echo mv ${name} ${new_name}

done
