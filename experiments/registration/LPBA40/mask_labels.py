import nibabel as nib
if __name__ == "__main__":
    lpba_base_dir = '/home/omar/data/LPBA40/delineation_space'
    for i in range(1,41):
        idx = '0'+str(i) if i<10 else str(i)
        strip_name = lpba_base_dir + '/S'+idx+'/S'+idx+'_strip.img'
        seg_name = lpba_base_dir + '/S'+idx+'/S'+idx+'_seg.img'
        print('Masking annotations: ' + seg_name)
        strip_ana = nib.load(strip_name)
        seg_ana = nib.load(seg_name)
        strip = strip_ana.get_data().squeeze()
        seg = seg_ana.get_data().squeeze()
        
        p = seg_name.find('_seg')
        strip_seg_name = seg_name[:p]+'_strip_seg.img'
        
        seg *= (strip>0)
        seg_ana.to_filename(strip_seg_name)

