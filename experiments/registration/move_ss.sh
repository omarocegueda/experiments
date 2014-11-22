for i in {1..18}
do
    if [ ${i} -ge 10 ];then
        idx=${i}
    else
        idx="0${i}"
    fi
    modality="pd"
    name=(`ls ${idx}/*${modality}*`)
    new_name="/home/omar/data/IBSR_nifti_stripped/IBSR_${idx}/IBSR_${idx}_ana_strip_${modality}.nii"
    cp ${name} ${new_name}
done
