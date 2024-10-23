from setuptools import setup

with open("requirements.txt", "r") as f:
    required_packages = f.read().splitlines()

setup(name='f110_gym',
      version='0.2.1',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      package_dir={'': 'gym'},
      install_requires=[*required_packages]
      )
