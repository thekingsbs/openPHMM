from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='openPHMM',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'scikit-learn>=1.0',
        'matplotlib>=3.4',
        'seaborn>=0.11',
        'tqdm>=4.62',
        'numba>=0.54',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'jupyter>=1.0',
            'black',
            'flake8',
        ]
    },
    author='Your Name',
    author_email='your@email.com',
    description='Profile Hidden Markov Models with discrete and Gaussian emissions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/openPHMM',
    python_requires='>=3.8',
)
