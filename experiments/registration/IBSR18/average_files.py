def average_files():
    import numpy as np
    import os
    import fnmatch
    ids = ['_segTRI_ana_', '_segTRI_fill_', '_seg_']

    for i, id in enumerate(ids):
        fnames=sorted([name for name in os.listdir(".") if fnmatch.fnmatch(name, 'jacard_IBSR_??'+id+'*.txt')])
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
        out = open('jacard_mean_'+str(i+1)+'.txt', 'w')
        out.writelines([str(s)+'\n' for s in means])

        out = open('jacard_std_'+str(i+1)+'.txt', 'w')
        out.writelines([str(s)+'\n' for s in stds])

if __name__ == '__main__':
    average_files()
