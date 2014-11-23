import os as _os
import numpy as np
import fnmatch
import experiments.registration.rcommon as rcommon

def get_labeling_info():
    '''
    labels, colors = get_labeling_info()
    '''
    dirname, base, ext = rcommon.decompose_path(__file__)
    common_labels_fname = dirname + 'common_labels.txt'

    with open(common_labels_fname) as f:
        lines=f.readlines()
    colors={}
    labels={}
    for line in lines:
        items=line.split()
        if not items:
            break
        colors[int(items[0])]=(float(items[2])/255.0, float(items[3])/255.0, float(items[4])/255.0)
        labels[int(items[0])]=items[1]
    return labels, colors

def average_files():
    labels, colors=getLabelingInfo()

    ids = ['_segTRI_ana_', '_segTRI_fill_', '_seg_']
    prefs = ['jaccard', 't_overlap']

    for pref in prefs:
        for i, id in enumerate(ids):
            fnames=sorted([name for name in os.listdir(".") if fnmatch.fnmatch(name, pref+'_IBSR_??'+id+'*.txt')])
            print(id, len(fnames))
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
            out = open(pref+'_mean_'+str(i+1)+'.txt', 'w')
            out.writelines([str(s)+'\n' for s in means])

            out = open(pref+'_std_'+str(i+1)+'.txt', 'w')
            out.writelines([str(s)+'\n' for s in stds])

            if(id == '_seg_'):
                scores = scores[:,[lab for lab in labels]]
                means = scores.mean(1)
                out = open(pref+'_boxplot_'+str(i+1)+'.txt', 'w')
                out.writelines([str(s)+'\n' for s in stds])

if __name__ == '__main__':
    average_files()
