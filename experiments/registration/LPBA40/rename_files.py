import subprocess
if __name__ == "__main__":
    lpba_base_dir = '/home/omar/data/LPBA40/delineation_space'
    for i in range(1,41):
        idx = '0'+str(i) if i<10 else str(i)
        strip_name = lpba_base_dir + '/S'+idx+'/S'+idx+'.delineation.skullstripped'
        seg_name = lpba_base_dir + '/S'+idx+'/S'+idx+'.delineation.structure.label'
        
        new_strip_name = lpba_base_dir + '/S'+idx+'/S'+idx+'_strip'
        new_seg_name = lpba_base_dir + '/S'+idx+'/S'+idx+'_seg'
        for ext in ['.img', '.hdr']:
            subprocess.call('mv '+strip_name+ext+' '+new_strip_name+ext, shell=True)
            subprocess.call('mv '+seg_name+ext+' '+new_seg_name+ext, shell=True)

