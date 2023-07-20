extern crate anyhow;
extern crate voxel_coarsen;

use voxel_coarsen::coarsen;

use std::fs::File;
use std::io::Write;

#[test]
fn read_file_data() {
    let file = String::from("/Users/carson16/Downloads/workflow_files-main/ornl_workflow_git/ExaConstit/pre_main_post_script/voxel_coarsen/data/0.2_ExaConstit.csv");
    let result = coarsen::voxel_coarsen(&file, 2);
    // let strings: Vec<String> = result.unwrap().iter().map(|n| n.to_string()).collect();

    // let file = File::create("/Users/carson16/Downloads/workflow_files-main/ornl_workflow_git/ExaConstit/pre_main_post_script/voxel_coarsen/data/coarse_vals.txt");
    // let _errs = writeln!(file.unwrap(), "{}", strings.join("\n"));
}
