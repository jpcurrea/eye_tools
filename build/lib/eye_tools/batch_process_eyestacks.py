#!/usr/bin/env python

"""Batch process all folders of images stacks and save focus stack.


Assumes the following folder structure of stacks of .jpg images:

.\
|--batch_process_stacks.py
|--eyestack_1\
   |--mask.jpg (optional: if absent, uses color selector GUI)
   |--img_001.jpg
   |--img_002.jpg
   |...
|--eyestack_1.jpg (outcome)
|--eyestack_2\
   |--mask.jpg (optional)
   |--img_001.jpg
   |--img_002.jpg
   |...
|--eyestack_2.jpg (outcome)
|--_hidden_folder\
   |(skipped files)
   |...
...
"""
import os
from scipy import misc
from analysis_tools import *

# load filenames and folders
fns = os.listdir(os.getcwd())
img_fns = [fn for fn in fns if fn.endswith(".jpg")]
folders = [fn for fn in fns if os.path.isdir(fn)]
folders = [os.path.join(os.getcwd(), f) for f in folders]
# for each folder
for folder in folders:
    # skip hidden folders
    base = os.path.basename(folder)
    if not base.startswith("_"):
        print(folder)
        # get stack name from the folder name
        path, base = os.path.split(folder)
        stack_name = "{}.jpg".format(base) # 
        # get mask image, if present
        fns = os.listdir(folder)
        fns = [os.path.join(folder, fn) for fn in fns]
        mask_fn = next([fn for fn in fns if "mask" in fn])
        # get the focus stack
        st = Stack(folder, f_type=".TIF")
        st.load()
        st.get_focus_stack()
        # save if the stack worked
        if st.stack is not None:
            new_fn = os.path.join(path, stack_name)
            plt.imsave(new_fn, st.stack.astype('uint8'))
        print()
