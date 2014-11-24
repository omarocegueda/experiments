for i in {1..18}
do
    if [ ${i} -ge 10 ];then
        iidx=${i}
    else
        iidx="0${i}"
    fi
    for j in {1..18}
    do
        if [ "${i}" != "${j}" ];then
            if [ ${j} -ge 10 ];then
                jidx=${j}
            else
                jidx="0${j}"
            fi
            name="IBSR_${iidx}_ana_strip_IBSR_${jidx}_ana_stripAffine.txt"
            t2t1name="IBSR_${iidx}_ana_strip_t2_IBSR_${jidx}_ana_stripAffine.txt"
            t1t2name="IBSR_${iidx}_ana_strip_IBSR_${jidx}_ana_strip_t2Affine.txt"
            pdt1name="IBSR_${iidx}_ana_strip_pd_IBSR_${jidx}_ana_stripAffine.txt"
            t1pdname="IBSR_${iidx}_ana_strip_IBSR_${jidx}_ana_strip_pdAffine.txt"
            pdt2name="IBSR_${iidx}_ana_strip_pd_IBSR_${jidx}_ana_strip_t2Affine.txt"
            t2pdname="IBSR_${iidx}_ana_strip_t2_IBSR_${jidx}_ana_strip_pdAffine.txt"
            cp ${name} ${t2t1name}
            cp ${name} ${t1t2name}
            cp ${name} ${pdt1name}
            cp ${name} ${t1pdname}
            cp ${name} ${pdt2name}
            cp ${name} ${t2pdname}
        fi
    done
done

