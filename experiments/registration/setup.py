from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("evaluation", ["evaluation.pyx"],include_dirs=get_numpy_include_dirs(), language="c"),
               Extension("splines", ["splines.pyx"],include_dirs=get_numpy_include_dirs(), language="c"),
               Extension("dti", ["dti.pyx"],include_dirs=get_numpy_include_dirs(), language="c"),
               Extension("cc_residuals", ["cc_residuals.pyx"],include_dirs=get_numpy_include_dirs(), language="c")]
setup(
  name = 'Evaluation tools',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
