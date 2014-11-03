import sys
import os
import fnmatch
import shutil
import subprocess
import errno

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


def split_all_pairs(names, shared_files):
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
            subprocess.call('ln '+target[0]+' '+dir_name+'/target', shell=True)
            subprocess.call('ln '+reference[0]+' '+dir_name+'/reference', shell=True)
            for f in required_files:
                subprocess.call('ln '+f+' '+dir_name, shell=True)
            for w in target[1:]:
                subprocess.call('ln '+w+' '+dir_name+'/warp', shell=True)
            with open(dir_name+'/jaccard_pairs.lst','w') as f:
                n = len(target)-1
                for k in range(n):
                    f.write(target[1+k]+' '+reference[0]+' '+reference[1+k]+'\n')


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
        dirName='0'+str(i+1) if i+1<10 else str(i+1)
        mkdir_p(os.path.join(dirName,'target'))
        mkdir_p(os.path.join(dirName,'reference'))
        mkdir_p(os.path.join(dirName,'warp'))
        subprocess.call('ln '+target+' '+dirName+'/target', shell=True)
        subprocess.call('ln '+reference+' '+dirName+'/reference', shell=True)

        for f in required_files:
            subprocess.call('ln '+f+' '+dir_name, shell=True)

        for w in names_moving[i][1:]:
            subprocess.call('ln '+w+' '+dirName+'/warp', shell=True)

def clean_working_dirs():
    dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
    for name in dirNames:
        shutil.rmtree(name)


def split_script(argv, required_files):
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
        st.clean_working_dirs()
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
        names=[line.strip().split() for line in lines]
        st.split_all_pairs(names, required_files)
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
        split_corresponding_pairs(names_moving, names_fixed, required_files)
        sys.exit(0)
    ############################Submit###################################
    if argv[1]=='u':
        dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
        for name in dirNames:
            os.chdir('./'+name)
            subprocess.call('qsub job*.sh -d .', shell=True)
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
