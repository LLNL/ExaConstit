from setuptools import setup
from setuptools_rust import Binding, RustExtension

import site
import sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="xtal_light_up",
    version="0.1",
    rust_extensions=[RustExtension("xtal_light_up.xtal_light_up", binding=Binding.PyO3,debug=False)],
    packages=["xtal_light_up"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)