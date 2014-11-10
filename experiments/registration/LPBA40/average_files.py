def average_files():
    import numpy as np
    import os
    import fnmatch
    prefs = ['jaccard', 't_overlap']

    for i, pref in enumerate(ids):
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
