import os as _os
import experiments.registration.rcommon as rcommon
_ibsr_base_dir = 'Unspecified'
_lpba_base_dir = 'Unspecified'
_brainweb_base_dir = 'Unspecified'

def _load_dataset_info():
    dirname, base, ext = rcommon.decompose_path(__file__)
    fname = dirname + base + '.txt'
    if _os.path.isfile(fname):
        with open(fname) as f:
            lines = [s.strip() for s in f.readlines()]
            if len(lines) != 3:
                print('Warning: expected base directories for IBSR, LPBA and Brainweb in '+fname+' in that order. Found '+str(len(lines))+' lines in file, you may get unexpected results')
            else:
                global _ibsr_base_dir
                global _lpba_base_dir
                global _brainweb_base_dir
                _ibsr_base_dir = lines[0]
                _lpba_base_dir = lines[1]
                _brainweb_base_dir = lines[2]
    else:
        print('Error: file not found. Expected base directories for IBSR, LPBA and Brainweb in text file "'+fname+'" in that order.')

_load_dataset_info()


def get_ibsr_base_dir():
    global _ibsr_base_dir
    return _ibsr_base_dir


def get_lpba_base_dir():
    global _lpba_base_dir
    return _lpba_base_dir


def get_brainweb_base_dir():
    global _brainweb_base_dir
    return _brainweb_base_dir


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
        fname = prefix + '_seg.img'
    elif data == 'strip':
        fname = prefix + '_strip.img'
    elif data == 'strip_seg':
        fname = prefix + '_strip_seg.img'
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


