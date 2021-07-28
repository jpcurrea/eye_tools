from setuptools import setup

setup(name="eye_tools",
      version='0.1.0',
      description='tools for image analysis, particularly for extracting\
      parameters relevant to the optical performance of compound eyes.',
      url="https://github.com/jpcurrea/eye_tools.git",
      author='Pablo Currea',
      author_email='johnpaulcurrea@gmail.com',
      license='MIT',
      packages=['eye_tools'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pandas',
          'pillow',
          'pyqt5',
          'scikit-image',
          'scikit-learn',
          'scipy',
      ],
      zip_safe=False)
