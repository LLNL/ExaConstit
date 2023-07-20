import numpy as np

import rust_voxel_coarsen.rust_voxel_coarsen as rvc

help(rvc)

file = "/Users/carson16/Downloads/workflow_files-main/ornl_workflow_git/ExaConstit/pre_main_post_script/voxel_coarsen/data/0.2_ExaConstit.csv"
file_comp = "/Users/carson16/Downloads/workflow_files-main/ornl_workflow_git/ExaConstit/pre_main_post_script/voxel_coarsen/data/coarse_vals.txt"

box_size, data = rvc.voxel_coarsen(file, 2)
data2 = np.loadtxt(file_comp, dtype=np.int32)

diff_data = data - data2

print(box_size)

print(np.sum(diff_data > 0))



