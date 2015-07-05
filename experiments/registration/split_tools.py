import sys
import os
import fnmatch
import shutil
import subprocess
import errno
from rcommon import decompose_path
import nibabel as nib
import numpy as np
from dipy.align.vector_fields import warp_3d_affine

def create_ref_correction_schedule(n, ref):
    regs = []
    centroid = ref
    paths = {}
    for i in range(n):
        if i == ref:
            paths[i] = [ref]
            continue
        regs.append((ref, i))
        paths[i] = [centroid, i]
    return regs, centroid, paths


def create_mst_correction_schedule(bvecs):
    # Create registration schedule
    n = bvecs.shape[0]
    A = np.abs(bvecs.dot(bvecs.T))
    in_set = np.zeros(n)
    set_sim = A[0,:].copy()
    in_set[0] = 1
    set_sim[0] = -1
    set_size = 1
    regs = []
    while(set_size < n):
        sel = np.argmax(set_sim)
        closest = -1
        for i in range(n):
            if in_set[i] == 0:
                set_sim[i] = max([set_sim[i], A[i, sel]])
            else:
                if closest == -1 or A[sel, i] > A[sel, closest]:
                    closest = i
        in_set[sel] = 1
        set_size += 1
        set_sim[sel] = -1
        regs.append((closest, sel))

    # Find the centroid
    A[...] = -1
    for i in range(n):
        A[i, i] = 0
    for reg in regs:
        A[reg[0], reg[1]] = 1
        A[reg[1], reg[0]] = 1

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if A[i, k] > 0 and A[k, j] > 0:
                    opt = A[i, k] + A[k, j]
                    if A[i, j] == -1 or opt < A[i, j]:
                        A[i, j] = opt
    steps = A.sum(1)
    centroid = np.argmin(steps)

    # Build correction schedule
    paths = {}
    paths[centroid] = [centroid]
    while(len(paths)<n):
        for reg in regs:
            if reg[0] in paths and reg[1] not in paths:
                paths[reg[1]] = paths[reg[0]] + [reg[1]]
            elif reg[1] in paths and reg[0] not in paths:
                paths[reg[0]] = paths[reg[1]] + [reg[0]]

    return regs, centroid, paths

def execute_path(path, matrices):
    n = len(path)
    affine = np.eye(4)
    for i in range(1,n):
        static = path[i-1]
        moving = path[i]
        new = matrices.get((static, moving))
        if new is None:
            new = matrices.get((moving, static))
            if new is None:
                raise ValueError("Path broken")
            new = np.linalg.inv(new)
        affine = affine.dot(new)
    return affine


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def link_image(imname, dir_name):
    subprocess.call('ln '+imname+' '+dir_name, shell=True)
    if imname[-4:]=='.img':
        hdr = imname[:-4] + '.hdr'
        subprocess.call('ln '+hdr+' '+dir_name, shell=True)

def split_all_pairs(names, required_files):
    r""" Creates a registration directory for each pair of images in names
    Each element of names is a list of file names.
    The first element of names[i] is the image to be registered, the rest
    are different image annotations (e.g. annotations according to tissue
    type, or according to anatomical region, etc.).
    For each pair of indices (i, j), i!=j, a working directory will be
    setup to register image names[i][0] against names[j][0], then theresulting
    warp will be applied to names[i][1..k] and the Jaccard index of each
    annotated region in each annotation image will be computed for each pair.
    For example, after registering

    reference=names[i][0]
    against
    target=names[j][0]

    the resulting warp is applied to names[j][1] using nearest-neighbor
    interpolation and the resulting warped annotations are compared against
    names[i][1], generating a Jaccard index for each region defined in
    names[i][1] and names[j][1]. Then the same warp is applied to names[j][2]
    and compared against names[i][2], and so on.

    """
    nlines=len(names)
    for i in range(nlines):
        if not names[i]:
            continue
        print 'Splitting reference:',names[i][0]
        reference=names[i]
        stri='0'+str(i+1) if i+1<10 else str(i+1)
        for j in range(nlines):
            if i==j:
                continue
            if not names[j]:
                continue
            target=names[j]
            strj='0'+str(j+1) if j+1<10 else str(j+1)

            dir_name=strj+'_'+stri
            mkdir_p(os.path.join(dir_name,'target'))
            mkdir_p(os.path.join(dir_name,'reference'))
            mkdir_p(os.path.join(dir_name,'warp'))
            link_image(target[0], dir_name+'/target')
            link_image(reference[0], dir_name+'/reference')
            for f in required_files:
                subprocess.call('ln '+f+' '+dir_name, shell=True)
            for w in target[1:]:
                link_image(w, dir_name+'/warp')
            with open(dir_name+'/jaccard_pairs.lst','w') as f:
                n = len(target)-1
                for k in range(n):
                    f.write(target[1+k]+' '+reference[0]+' '+reference[1+k]+'\n')


def split_indirect_validation(labeled, unlabeled, required_files):
    r"""
    Register all pairs (a, b) from 'labeled' to each c in 'unlabeled'. The
    score will be computed by composing the transformations a-->c and c-->b
    and measuring the Jaccard index of regions in 'a' with those of 'b' after
    deformation.
    """
    nlabeled = len(labeled)
    nunlabeled = len(unlabeled)

    for i in range(nlabeled):
        stri='0'+str(i+1) if i+1<10 else str(i+1)
        for j in range(nlabeled):
            if i == j:
                continue
            # For each pair in labeled
            strj='0'+str(j+1) if j+1<10 else str(j+1)


            for k in range(nunlabeled):
                # For each unlabeled
                strk='0'+str(k+1) if j+1<10 else str(k+1)

                dir_name=strj+'_'+stri+'_'+strk

                mkdir_p(os.path.join(dir_name,'ref_fixed'))
                mkdir_p(os.path.join(dir_name,'ref_moving'))
                mkdir_p(os.path.join(dir_name,'target'))
                mkdir_p(os.path.join(dir_name,'warp'))

                link_image(labeled[i][0], dir_name+'/ref_fixed')
                link_image(labeled[j][0], dir_name+'/ref_moving')
                link_image(unlabeled[k], dir_name+'/target')

                for f in required_files:
                    subprocess.call('ln '+f+' '+dir_name, shell=True)

                for w in labeled[j][1:]:
                    link_image(w, dir_name+'/warp')

                with open(dir_name+'/jaccard_pairs.lst','w') as f:
                    n = len(labeled[j]) - 1
                    for h in range(n):
                        f.write(labeled[j][1 + h]+' '+labeled[i][0]+' '+labeled[i][1 + h]+'\n')




def split_all_pairs_multi_modal(names, required_files, mod1='', mod2='_t2'):
    r""" Creates a registration directory for each pair of images in names
    The suffix mod1 and mod2 are added to each file name before the file extension
    (we assume that semi-synthetic images are named the same as the original anatomy
    file plus a suffix indicating the modality).
    Each element of names is a list of file names.
    The first element of names[i] is the image to be registered, the rest
    are different image annotations (e.g. annotations according to tissue
    type, or according to anatomical region, etc.).
    For each pair of indices (i, j), i!=j, a working directory will be
    setup to register image names[i][0] against names[j][0], then the resulting
    warp will be applied to names[i][1..k] and the Jaccard index of each
    annotated region in each annotation image will be computed for each pair.
    For example, after registering

    reference=names[i][0]
    against
    target=names[j][0]

    the resulting warp is applied to names[j][1] using nearest-neighbor
    interpolation and the resulting warped annotations are compared against
    names[i][1], generating a Jaccard index for each region defined in
    names[i][1] and names[j][1]. Then the same warp is applied to names[j][2]
    and compared against names[i][2], and so on.

    """
    nlines=len(names)
    for i in range(nlines):
        if not names[i]:
            continue
        print 'Splitting reference:',names[i][0]
        reference=names[i]
        stri='0'+str(i+1) if i+1<10 else str(i+1)
        for j in range(nlines):
            if i==j:
                continue
            if not names[j]:
                continue
            target=names[j]
            strj='0'+str(j+1) if j+1<10 else str(j+1)

            #######target mod1 vs reference mod2########
            if mod1 != mod2:
                dir_name=strj+'_'+stri+'_mod1_to_mod2'
            else:
                dir_name=strj+'_'+stri
            mkdir_p(os.path.join(dir_name,'target'))
            mkdir_p(os.path.join(dir_name,'reference'))
            mkdir_p(os.path.join(dir_name,'warp'))

            dir, name, ext = decompose_path(target[0])
            effective_target = dir+name+mod1+ext

            dir, name, ext = decompose_path(reference[0])
            effective_reference = dir+name+mod2+ext

            link_image(effective_target, dir_name+'/target')
            link_image(effective_reference, dir_name+'/reference')
            for f in required_files:
                subprocess.call('ln '+f+' '+dir_name, shell=True)
            for w in target[1:]:
                link_image(w, dir_name+'/warp')
            with open(dir_name+'/jaccard_pairs.lst','w') as f:
                n = len(target)-1
                for k in range(n):
                    f.write(target[1+k]+' '+effective_reference+' '+reference[1+k]+'\n')

            #######target mod2 vs reference mod1########
            if mod1 == mod2:
                continue
            dir_name=strj+'_'+stri+'_mod2_to_mod1'
            mkdir_p(os.path.join(dir_name,'target'))
            mkdir_p(os.path.join(dir_name,'reference'))
            mkdir_p(os.path.join(dir_name,'warp'))

            dir, name, ext = decompose_path(target[0])
            effective_target = dir+name+mod2+ext

            dir, name, ext = decompose_path(reference[0])
            effective_reference = dir+name+mod1+ext

            link_image(effective_target, dir_name+'/target')
            link_image(effective_reference, dir_name+'/reference')
            for f in required_files:
                subprocess.call('ln '+f+' '+dir_name, shell=True)
            for w in target[1:]:
                link_image(w, dir_name+'/warp')
            with open(dir_name+'/jaccard_pairs.lst','w') as f:
                n = len(target)-1
                for k in range(n):
                    f.write(target[1+k]+' '+effective_reference+' '+reference[1+k]+'\n')


def split_corresponding_pairs(names_moving, names_fixed, required_files):
    r""" Creates a registration directory for each specified pair of images

    """
    nlines_moving=len(names_moving)
    nlines_fixed=len(names_fixed)
    if nlines_fixed!=nlines_moving:
        print 'Error: the number of files in the moving (%d) list is not the same as in the fixed list(%d)'%(nlines_moving, nlines_fixed)
        sys.exit(0)
    for i in range(nlines_moving):
        target=names_moving[i][0]
        reference=names_fixed[i][0]
        if (not target) or (not reference):
            continue
        print 'Generating registration folder:', target, reference
        dir_name='0'+str(i+1) if i+1<10 else str(i+1)
        mkdir_p(os.path.join(dir_name,'target'))
        mkdir_p(os.path.join(dir_name,'reference'))
        mkdir_p(os.path.join(dir_name,'warp'))
        link_image(target, dir_name+'/target')
        link_image(reference, dir_name+'/reference')

        for f in required_files:
            subprocess.call('ln '+f+' '+dir_name, shell=True)

        for w in names_moving[i][1:]:
            link_image(w, dir_name+'/warp')

def clean_working_dirs():
    dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
    for name in dirNames:
        shutil.rmtree(name)


def split_script(argv, required_files, task_type='mono', mod1="", mod2=""):
    argc=len(argv)
    #############################No parameters#############################
    if not argv[1]:
        print 'Please specify an action: c "(clean)", s "(split)", u "(submit)", o "(collect)"'
        sys.exit(0)
    #############################Clean#####################################
    if argv[1]=='c':
        if not os.path.isdir('results'):
            cleanAnyway=query_yes_no("It seems like you have not collected the results yet. Clean anyway? (y/n)")
            if not cleanAnyway:
                sys.exit(0)
        clean_working_dirs()
        sys.exit(0)
    #############################Split####################################
    if argv[1]=='s':
        if argc<3:
            print 'Please specify a text file containing the names of the files to register'
            sys.exit(0)
        try:
            with open(argv[2]) as f:
                lines=f.readlines()
        except IOError:
            print 'Could not open file:', argv[2]
            sys.exit(0)
        if not os.path.isdir('affine'):
            warning = "WARNING: this directory does not contain an 'affine' folder."
            warning += "Precomputed affine transforms should be located in that directory,"
            warning += "otherwise all affine registrations will be run. (If you have already precomputed"
            warning += "the affine transforms, you may use 'ln -s' to create a symlink here to the folder"
            warning += "that contains the results)"
            print(warning)
        names=[line.strip().split() for line in lines]
        if(task_type=='multi'):
            split_all_pairs_multi_modal(names, required_files, mod1, mod2)
        elif(task_type=='mono'):
            split_all_pairs(names, required_files)
        elif(task_type=='indirect'):
            unlabeled = []
            with open(argv[3]) as f:
                unlabeled=[line.strip() for line in f.readlines()]
            split_indirect_validation(names, unlabeled, required_files)
        else:
            print("Unknown task type: "+task_type)
        sys.exit(0)
    if argv[1]=='s2':#provide two file lists: moving and fixed
        if argc<4:
            print 'Please specify two text files containing the names of the moving and fixed images to register'
            sys.exit(0)
        try:
            with open(argv[2]) as f:
                linesMoving=f.readlines()
        except IOError:
            print 'Could not open file:', argv[2]
            sys.exit(0)
        try:
            with open(argv[3]) as f:
                linesFixed=f.readlines()
        except IOError:
            print 'Could not open file:', argv[3]
            sys.exit(0)
        names_moving=[line.strip().split() for line in linesMoving]
        names_fixed=[line.strip().split() for line in linesFixed]
        if(task_type=='mono'):
            split_corresponding_pairs(names_moving, names_fixed, required_files)
        sys.exit(0)
    ############################Submit###################################
    if argv[1]=='u':
        dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
        for name in dirNames:
            os.chdir('./'+name)
            subprocess.call('qsub job*.sh -d . -q batch', shell=True)
            os.chdir('./..')
        sys.exit(0)
    ############################Collect##################################
    if argv[1]=='o':
        mkdir_p('results')
        dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
        for name in dirNames:
            subprocess.call('mv '+os.path.join(name,'*.nii.gz')+' results', shell=True)
            subprocess.call('mv '+os.path.join(name,'*.txt')+' results', shell=True)
            subprocess.call('mv '+os.path.join(name,'*.e*')+' results', shell=True)
            subprocess.call('mv '+os.path.join(name,'*.o*')+' results', shell=True)
        sys.exit(0)
    ############################Unknown##################################
    print 'Unknown option "'+argv[1]+'". The available options are "(c)"lean, "(s)"plit, s"(u)"bmit, c"(o)"llect.'


def split_dwi(argv, required_files):
    argc=len(argv)
    #############################No parameters#############################
    if argc < 2:
        print 'Please specify an action: c "(clean)", s "(split)", u "(submit)", o "(collect)", e "(execute)"'
        sys.exit(0)
    #############################Clean#####################################
    if argv[1]=='c':
        if not os.path.isdir('results'):
            cleanAnyway=query_yes_no("It seems like you have not collected the results yet. Clean anyway? (y/n)")
            if not cleanAnyway:
                sys.exit(0)
        clean_working_dirs()
        sys.exit(0)
    #############################Split#####################################
    if argv[1]=='s':
        if argc < 3:
            print('Please specify the dwi file name')
            sys.exit(0)
        dwi_fname = argv[2]
        dwi_nib = nib.load(dwi_fname)
        dwi = dwi_nib.get_data().squeeze()
        n = dwi.shape[3]
        print('Loaded %d volumes' % (n))
        if argc < 4:
            print('Please specify the reference volume or schedule type')
            sys.exit(0)
        try:
            reference = int(argv[3])
            if reference < 0 or reference >= n:
                print('Invalid reference volume: %d' % (reference))
                sys.exit(0)
            regs, centroid, paths = create_ref_correction_schedule(n, reference)
            print('Registration schedule (star) with centroid %d.' % (centroid,))
        except:
            if argv[3] != 'MST':
                print('Undefined schedule: '+ argv[3])
                sys.exit(0)
            if argc < 5:
                print('Please provide the (n x 4) B-matrix file name.')
                sys.exit(0)
            Bfname = argv[4]
            B = np.loadtxt(Bfname)
            if B.shape != (n,4):
                print('Invalid B-matrix. Shape is (%d, %d). Expected: (%d, %d)' % (B.shape[0], B.shape[1], n, 4))
                sys.exit(0)
            regs, centroid, paths = create_mst_correction_schedule(B[:,:3])
            print('Registration schedule (mst) with centroid %d.' % (centroid,))

        # Create one image per volume
        mkdir_p('dwi_split')
        print('Creating individual volumes...')
        for i in range(n):
            fname = os.path.join('dwi_split', 'dwi_%03d.nii.gz' % (i,))
            print('Extracting volume %s ...' % fname)
            i_nib = nib.Nifti1Image(dwi[...,i], dwi_nib.get_affine())
            i_nib.to_filename(fname)
        # Create the registration jobs
        print('Creating registration jobs...')
        for reg in regs:
            i, j = reg
            stri = '%02d' % (i,)
            strj = '%02d' % (j,)
            ifname = os.path.join('dwi_split', 'dwi_%03d.nii.gz' % (i,))
            jfname = os.path.join('dwi_split', 'dwi_%03d.nii.gz' % (j,))
            dir_name = strj + '_' + stri
            target_path = os.path.join(dir_name, 'target')
            reference_path = os.path.join(dir_name, 'reference')
            mkdir_p(os.path.join(dir_name,'target'))
            mkdir_p(os.path.join(dir_name,'reference'))
            mkdir_p(os.path.join(dir_name,'warp'))
            subprocess.call('ln %s %s' % (jfname, target_path), shell=True)
            subprocess.call('ln %s %s' % (ifname, reference_path), shell=True)
            for f in required_files:
                subprocess.call('ln %s %s' % (f, dir_name), shell=True)
        sys.exit(0)
    if argv[1]=='u':
        dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
        for name in dirNames:
            os.chdir('./'+name)
            subprocess.call('qsub job*.sh -d . -q batch', shell=True)
            os.chdir('./..')
        sys.exit(0)
    ############################Collect##################################
    if argv[1]=='o':
        mkdir_p('results')
        dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
        for name in dirNames:
            subprocess.call('mv '+os.path.join(name,'*.txt')+' results', shell=True)
            subprocess.call('mv '+os.path.join(name,'*.e*')+' results', shell=True)
            subprocess.call('mv '+os.path.join(name,'*.o*')+' results', shell=True)
        sys.exit(0)
    ############################Execute##################################
    if argv[1]=='e':
        if argc < 3:
            print('Please specify the dwi file name')
            sys.exit(0)
        dwi_fname = argv[2]
        dwi_nib = nib.load(dwi_fname)
        dwi = dwi_nib.get_data().squeeze()
        n = dwi.shape[3]
        print('Loaded %d volumes' % (n))
        if argc < 4:
            print('Please specify the reference volume or schedule type')
            sys.exit(0)
        try:
            reference = int(argv[3])
            if reference < 0 or reference >= n:
                print('Invalid reference volume: %d' % (reference))
                sys.exit(0)
            regs, centroid, paths = create_ref_correction_schedule(n, reference)
            destination = 'star_%d' % reference
            print('Registration schedule (star) with centroid %d.' % (centroid,))
        except:
            if argv[3] != 'MST':
                print('Undefined schedule: '+ argv[3])
                sys.exit(0)
            if argc < 5:
                print('Please provide the B-matrix file name.')
                sys.exit(0)
            Bfname = argv[4]
            B = np.loadtxt(Bfname)
            n = B.shape[0]
            regs, centroid, paths = create_mst_correction_schedule(B[:,:3])
            destination = 'mst_%d' % reference
            print('Registration schedule (mst) with centroid %d.' % (centroid,))

        mkdir_p(destination)
        matrices = {}
        for reg in regs:
            i, j = reg
            mname = 'dwi_%03d_dwi_%03dAffine.txt' % (j, i)
            matrices[(i,j)] = np.loadtxt(mname)
            print('Loaded matrix %s' % mname)
        corrected = np.empty_like(dwi)
        for i in range(n):
            affine = execute_path(paths[i], matrices)
            # Warp image i
            M = np.linalg.inv(dwi_nib.get_affine())
            M = M.dot(affine)
            M = M.dot(dwi_nib.get_affine())

            in_vol = dwi[...,i].astype(np.float32)
            out_shape = np.array(dwi[...,i].shape, dtype=np.int32)
            warped = warp_3d_affine(in_vol, out_shape, M)
            corrected[...,i] = warped[...]
            affname = os.path.join(destination, 'dwi_%03d_dwi_%03dAffine.txt' % (i, centroid))
            np.savetxt(affname, affine)
        corrected_nib = nib.Nifti1Image(corrected, dwi_nib.get_affine())
        oname = os.path.join(destination, 'corrected.nii.gz')
        corrected_nib.to_filename(oname)
        sys.exit(0)
    ############################Unknown##################################
    print 'Unknown option "'+argv[1]+'". The available options are "(c)"lean, "(s)"plit, s"(u)"bmit, c"(o)"llect.'













