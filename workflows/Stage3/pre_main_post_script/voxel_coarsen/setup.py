from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="rust_voxel_coarsen",
    version="0.1",
    rust_extensions=[RustExtension("rust_voxel_coarsen.rust_voxel_coarsen", binding=Binding.PyO3, debug=False,)],
    packages=["rust_voxel_coarsen"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)