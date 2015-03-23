import experiments.registration.dataset_info as info
if __name__ == "__main__":
    scil_base_dir = info.get_scil_base_dir()
    with open('names_scil.txt','w') as f:
        for i in range(1,2):
            idx = '0'+str(i) if i<10 else str(i)
            strip_name = scil_base_dir + '/SCIL_'+idx+'/SCIL_'+idx+'_b0_down_strip.nii.gz'
            f.write(strip_name+'\n')
