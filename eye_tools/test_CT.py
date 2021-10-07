"""Test script for processing a CT stack.
"""
from analysis_tools import *


PIXEL_SIZE = 4.50074
DEPTH_SIZE = PIXEL_SIZE

ct_directory = "/home/pbl/Downloads/moth_eye/"
img_ext = ".tif"
# make a ct stack object using the ct stack folder
# ct_stack = CTStack(dirname=ct_directory, img_extension=img_ext, bw=True,
#                    pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
# ct_stack.prefilter(low=5000)
# make a CT stack using the prefiltered_stack
# ct_directory = "../../../research/method__compound_eye_tools/data/CT/manuscript/manduca_sexta/_prefiltered_stack"
# make a ct stack object using the ct stack folder
ct_stack = CTStack(dirname=ct_directory, img_extension=img_ext, bw=True,
                   pixel_size=PIXEL_SIZE, depth_size=DEPTH_SIZE)
ct_stack.ommatidia_detecting_algorithm(
    polar_clustering=True, display=True, test=True)
