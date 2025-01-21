from setuptools import setup, find_packages

setup(
    name='ECTMetrics',
    version='0.1.0',
    author='Max Kayser',
    author_email='max.kayser@uni-bonn.de',
    description='A package for EEG signal processing and ECT seizure quality metrics calculation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/maxkayser/ECTMetrics',
    packages=find_packages(),
    #package_dir={"": "ectmetrics"},
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'pyedflib'
    ],
    #tests_require=[
    #    'pytest>=6.0.0',
    #],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    python_requires='>=3.6',
    keywords='ECT, electroconvulsive therapy, seizure, metrics, signal processing, seizure analysis, EEG, electroencephalography, suppression, medical, biomedical',
)
