from setuptools import setup, find_packages

setup(
    name='small-transformer',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'tqdm>=4.65.0',
        'scikit-learn>=1.2.0',
    ],
    author='Transformer Developer',
    description='A small Transformer model for NLP tasks on CPU',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)