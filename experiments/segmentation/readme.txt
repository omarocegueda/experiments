0. Make sure to add the experiments folder to your python path, for example in your ~/.bashrc or ~/.bash_profile (of course replace with the location of your code):
    export PYTHONPATH=$PYTHONPATH:/home/omar/opt/experiments

1. First you need to compile the evaluation module, this is the cython module that contains the jaccard-index computation code and a baseline segmentation algorithm (k-means)
    a) From the `segmentation` folder run:
        python setup.py build_ext --inplace

2. It will be useful to create a folder for each experiment you want to run. Inside the `segmentation` folder, create an `IBSR` sub-folder

3. Now you need to create a text file called that contains the file-names of the IBSR data set, both the input images and the ground truth segmentations. For example, 
   my names_ibsr_seg.txt file looks like:
/home/omar/data/IBSR_nifti_stripped/IBSR_02/IBSR_02_ana_strip.nii.gz    /home/omar/data/IBSR_nifti_stripped/IBSR_02/IBSR_02_segTRI_fill_ana.nii.gz
/home/omar/data/IBSR_nifti_stripped/IBSR_03/IBSR_03_ana_strip.nii.gz    /home/omar/data/IBSR_nifti_stripped/IBSR_03/IBSR_03_segTRI_fill_ana.nii.gz
/home/omar/data/IBSR_nifti_stripped/IBSR_04/IBSR_04_ana_strip.nii.gz    /home/omar/data/IBSR_nifti_stripped/IBSR_04/IBSR_04_segTRI_fill_ana.nii.gz
...
name this file `names_ibsr_seg.txt` and put it inside your `IBSR` folder

4. We are now ready to split, submit and collect the segmentation jobs/results:
    a) Split the jobs by running (from your `IBSR` sub-folder)
        python ../split_baseline.py s names_ibsr_seg.txt
       this will create one folder for each ibsr image and copy the cluster job (a shell script) to segment the image with the baseline algorithm
    b) Submit your jobs by running (still, from your `IBSR` subfolder)
        python ../split_baseline.py u
    c) Wait until your jobs have finished. A few useful commands to check the status:
        qstat -q
        qstat | less
        qstat | grep " R " |less
    d) Collect the results and put them inside a single folder by running:
        python ../split_baseline.py o
       The `results` subfolder now contains all the segmented files and a text file with the jaccard indices for each segmentation
    e) Once you have moved the files from the `results` folder to a safe location, clean your working directory:
        python ../split_baseline.py c
