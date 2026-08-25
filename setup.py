from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    requirements_path = Path(__file__).with_name('requirements.txt')
    return [
        line.strip()
        for line in requirements_path.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]

setup(
    name='small-transformer',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=read_requirements(),
    author='Transformer Developer',
    description='A small Transformer model for NLP tasks on CPU',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
