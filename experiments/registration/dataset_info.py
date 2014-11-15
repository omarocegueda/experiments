

def get_ibsr_base_dir():
    return '/home/omar/data/IBSR_nifti_stripped/'


def get_lpba_base_dir():
    return '/home/omar/data/LPBA40/delineation_space/'


def get_brainweb_base_dir():
    return '/home/omar/data/Brainweb/'


def get_ibsr(idx, data):
    ibsr_base_dir = get_ibsr_base_dir()
    if idx<10:
        idx = '0'+str(idx)
    else:
        idx = str(idx)
    prefix = ibsr_base_dir + 'IBSR_'+idx+'/IBSR_'+idx
    fname = None
    if data == 'mask':
        fname = prefix + '_ana_brainmask.nii.gz'
    elif data == 'seg3':
        fname = prefix + '_segTRI_fill_ana.nii.gz'
    elif data == 'seg':
        fname = prefix + '_seg_ana.nii.gz'
    elif data == 'raw':
        fname = prefix + '_ana.nii.gz'
    elif data == 'strip':
        fname = prefix + '_ana_strip.nii.gz'
    elif data == 't1':
        fname = prefix + '_ana_strip.nii.gz'
    elif data == 't2':
        fname = prefix + '_ana_strip_t2.nii.gz'
    return fname


def get_lpba(idx, data):
    lpba_base_dir = get_lpba_base_dir()
    if idx<10:
        idx = '0'+str(idx)
    else:
        idx = str(idx)
    prefix = lpba_base_dir + 'S'+idx+'/S'+idx
    fname = None
    if data == 'seg':
        fname = prefix + '_seg.nii.gz'
    elif data == 'strip':
        fname = prefix + '_strip.nii.gz'
    elif data == 'strip_seg':
        fname = prefix + '_strip_seg.nii.gz'
    return fname


def get_brainweb(modality, data):
    if not modality in ['t1', 't2', 'pd']:
        return None
    modality = modality.lower()
    brainweb_dir = get_brainweb_base_dir()+modality
    fname = None
    if data is 'strip':
        fname = brainweb_dir+'/brainweb_'+modality+'_strip.nii.gz'
    elif data is 'raw':
        fname = brainweb_dir+'/brainweb_'+modality+'.nii.gz'
    return fname


