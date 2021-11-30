from setuptools import setup

setup(name="eye_tools",
      version='0.2.1',
      description='tools for image analysis, particularly for extracting\
      parameters relevant to the optical performance of compound eyes.',
      url="https://github.com/jpcurrea/compound_eye_tools.git",
      author='Pablo Currea',
      author_email='johnpaulcurrea@gmail.com',
      license='MIT',
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'scikit-image',
          'pandas',
          'pillow',
          'h5py',
          'pyqtgraph',
          'PyQt5'
      ],
      zip_safe=False)
