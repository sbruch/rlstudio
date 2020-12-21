from setuptools import find_namespace_packages
from setuptools import setup

setup(
    name='rlstudio',
    version='0.1',
    url='https://github.com/sbruch/rlstudio',
    author='sbruch',
    description=('An RL framework.'),
    install_requires=[
        'absl-py>=0.11.0',
        'jax>=0.2.5',
        'jaxlib>=0.1.57',
        'dm-env>=1.3',
        'numpy>=1.19.4',
        'rlax>=0.0.2',
        'optax>=0.0.1',
        'dm-haiku>=0.0.2',
    ],
    license='MIT',
    packages=find_namespace_packages(include=['rlstudio.*']),
    python_requires='>=3.6',
)
