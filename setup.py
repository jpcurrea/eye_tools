from setuptools import setup

setup(name="eye_tools",
      version='0.1.1',
      description='tools for image analysis, particularly for extracting\
      parameters relevant to the optical performance of compound eyes.',
      url="https://github.com/jpcurrea/compound_eye_tools.git",
      author='Pablo Currea',
      author_email='johnpaulcurrea@gmail.com',
      license='MIT',
      packages=['fly_eye'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'scikit-image',
          'opencv-python',
          'pandas',
          'pillow',
          'pyqt5',
      ],
      zip_safe=False)
