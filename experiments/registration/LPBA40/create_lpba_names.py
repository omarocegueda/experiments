import experiments.registration.dataset_info as info
if __name__ == "__main__":
    lpba_base_dir = info.get_lpba_base_dir()
    with open('names_lpba_full.txt','w') as f:
        for i in range(1,41):
            idx = '0'+str(i) if i<10 else str(i)
            strip_name = lpba_base_dir + '/S'+idx+'/S'+idx+'_strip.img'
            seg_name = lpba_base_dir + '/S'+idx+'/S'+idx+'_strip_seg.img'
            f.write('\t'.join([strip_name, seg_name, '\n']))

