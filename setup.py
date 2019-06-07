"""Installation script."""
from setuptools import setup, find_packages

setup(
    name='ecodse-funtime-alpha',
    python_requires='>=3.6',
    version='0.1a',
    description='Jag for MRQA',
    long_description='Move like jagger',
    url='https://github.com/jerpint/jag',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7'
    ],
    packages=find_packages(exclude=['docs', 'tests']),

    install_requires=['tensorflow==1.13.1',
                      'numpy',
                      'pandas',
                      'boto3',
                      'tqdm',
                      'mlflow'],

    extras_require={'dev': ['flake8', 'pytest']}
)
