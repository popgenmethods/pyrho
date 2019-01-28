from pyrho import VERSION
from distutils.core import setup
setup(
    name='pyrho',
    version=VERSION,
    description='Inference of fine-scale recombination maps using composite '
                'likelihoods and fused-LASSO.',
    author='Jeffrey P. Spence, Yun S. Song',
    author_email='spence.jeffrey@berkeley.edu',
    packages=['pyrho'],
    install_requires=['numpy>=1.14.2', 'scipy>=0.19.0', 'msprime>=0.4.0',
                      'numba>=0.42.0', 'pandas>=0.23.4', 'tables>=3.3.0',
                      'cyvcf2', 'ldpop'],
    package_data={'pyrho': ['data/pyrho_hapmap_maps.txt']},
    entry_points={
        'console_scripts': ['pyrho = pyrho.frontend:main'],
    },
)
