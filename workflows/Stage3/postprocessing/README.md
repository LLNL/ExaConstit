# Postprocessing

Prerequisites:
- Python3 is needed to run the post processing script
- The simulations need to have been run so we have the necessary volume average/integrated values to use such as the stress, plastic work, or plastic deformation rate tensor.

This script takes in the general output file directory, unique rve name, temperatures, and an output directory file name. It will then run an optimization script to determine a set of parameters that best optimize the Barlat Yld2004-18p parameters based on the results from the simulations. I have not gotten a chance to actually run the cli interface part of things. So, a few bugs may exist. The optimization script and all the other components related to it have been tested though.