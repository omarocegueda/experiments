from experiments.registration.dipyreg import compute_jaccard, compute_target_overlap
from experiments.registration.dataset_info import get_lpba_base_dir
import os
def compute_all_scores():
    lpba_dir = get_lpba_base_dir()
    cnt = 0
    for i in range(1,41):
        i_idx = str(i) if i>=10 else '0'+str(i)
        for j in range(1,41):
            if i == j:
                continue
            cnt += 1
            j_idx = str(j) if j>=10 else '0'+str(j)
            reference_name = lpba_dir+'/S'+i_idx+'/S'+i_idx+'_strip_seg.img'
            warped_name = 'warpedDiff_S'+j_idx+'_strip_seg_S'+i_idx+'_strip.nii.gz'
            if os.path.exists(reference_name) and os.path.exists(warped_name):
                print(str(cnt)+'/'+str(40*39), reference_name, warped_name)
                compute_target_overlap(reference_name, warped_name, True)
                compute_jaccard(reference_name, warped_name, True)


def average_files():
    import numpy as np
    import os
    import fnmatch
    prefs = ['jaccard', 't_overlap']

    for i, pref in enumerate(prefs):
        fnames=sorted([name for name in os.listdir(".") if fnmatch.fnmatch(name, pref+'_S*.txt')])
        print(pref, len(fnames))
        scores = []
        max_len = 0
        for name in fnames:
            with open(name, 'r') as input:
                line_scores = np.array([float(line) for line in input.readlines()])
                max_len = max([max_len, len(line_scores)])
                scores.append(line_scores)
        
        for j, line_scores in enumerate(scores):
            n = line_scores.shape[0]
            new_scores = np.ndarray(max_len, dtype=np.double)
            new_scores[:n] = line_scores[...]
            scores[j] = new_scores

        scores = np.array(scores)
        print(scores.shape)
        means = scores.mean(0)
        stds = scores.std(0)
        print(means)   
        out = open(pref+'_mean.txt', 'w')
        out.writelines([str(s)+'\n' for s in means])

        out = open(pref+'_std.txt', 'w')
        out.writelines([str(s)+'\n' for s in stds])

if __name__ == '__main__':
    average_files()
