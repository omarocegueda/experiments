import sys
import os
import fnmatch
import shutil
import subprocess
import errno
from rcommon import decompose_path
import nibabel as nib
import numpy as np


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

def split_all_images(names, required_files):
    r""" Creates a registration directory for each image to be segmented.

    Each element of names is a list of file names.
    The first element of names[i] is the image to be segmented, the second
    is the ground truth segmentation (e.g. annotations according to tissue
    type). For each image a working directory will be setup to segment and
    evaluate that image by computing the Jaccard index of each tissue type.
    """
    nlines=len(names)
    for i in range(1, 1 + nlines):
        print('Creating working directory for image: %s' % (names[i-1][0],))
        current_set = names[i-1]
        dir_name = "%02d" % (i)

        mkdir_p(os.path.join(dir_name,'input'))
        mkdir_p(os.path.join(dir_name,'seg'))
        link_image(current_set[0], dir_name+'/input')
        link_image(current_set[1], dir_name+'/seg')
        for f in required_files:
            subprocess.call('ln '+f+' '+dir_name, shell=True)


def clean_working_dirs():
    dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
    for name in dirNames:
        shutil.rmtree(name)


def split_script(argv, required_files, task_type='mono', mod1="", mod2=""):
    ''' The main script parsing and executing command line instructions
    '''
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
            print 'Please specify a text file containing the names of the files to segment'
            sys.exit(0)
        try:
            with open(argv[2]) as f:
                lines=f.readlines()
        except IOError:
            print 'Could not open file:', argv[2]
            sys.exit(0)
        names=[line.strip().split() for line in lines]
        split_all_images(names, required_files)
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


def create_ibsr_names(base_dir):
    with open('names_ibsr_seg.txt','w') as fout:
        for i in range(1, 19):
            img_fname = os.path.join(base_dir, 'IBSR_%02d' % (i), 'IBSR_%02d_ana_strip.nii.gz' % (i))
            seg_fname = os.path.join(base_dir, 'IBSR_%02d' % (i), 'IBSR_%02d_segTRI_fill_ana.nii.gz' % (i))
            fout.write(img_fname + '\t' + seg_fname + '\n')
