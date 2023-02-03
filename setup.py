from setuptools import setup, find_packages

setup(
  name = 'gigagan-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'GigaGAN - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/ETSformer-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'generative adversarial networks'
  ],
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.4',
    'open-clip-torch>=2.0.0,<3.0.0',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: 