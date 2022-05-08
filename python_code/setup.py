from setuptools import setup

setup(name='gym_packman',
  version='1.0',
  install_requires=['gym==0.21.0', 'tensorflow==2.6', 'matplotlib', 'IPython', 'opencv-python'],  # And any other dependencies Env needs
  packages=["gym_packman"]
  )