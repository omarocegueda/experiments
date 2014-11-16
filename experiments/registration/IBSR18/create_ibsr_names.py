import experiments.registration.dataset_info as info
if __name__ == "__main__":
    ibsr_base_dir = info.get_ibsr_base_dir()
    with open('names_ibsr.txt','w') as f:
        for i in range(1,19):
            idx = '0'+str(i) if i<10 else str(i)
            strip_name = ibsr_base_dir + '/IBSR_'+idx+'/IBSR_'+idx+'_ana_strip.nii.gz'
            f.write(strip_name+'\n')

    with open('names_ibsr_full.txt','w') as f:
        for i in range(1,19):
            idx = '0'+str(i) if i<10 else str(i)
            strip_name = ibsr_base_dir + '/IBSR_'+idx+'/IBSR_'+idx+'_ana_strip.nii.gz'
            segTri_fill_name = ibsr_base_dir + '/IBSR_'+idx+'/IBSR_'+idx+'_segTRI_fill_ana.nii.gz'
            segTri_name = ibsr_base_dir + '/IBSR_'+idx+'/IBSR_'+idx+'_segTRI_ana.nii.gz'
            seg_name = ibsr_base_dir + '/IBSR_'+idx+'/IBSR_'+idx+'_seg_ana.nii.gz'
            f.write('\t'.join([strip_name, segTri_fill_name, segTri_name, seg_name, '\n']))
