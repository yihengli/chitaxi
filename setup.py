from setuptools import setup, find_packages

setup(
    name='chitaxi',
    version='0.1.0',
    description='a library for chicago taxi analysis',
    author='Yiheng Li, Huiyan Xu, Pingchuan Ma, Zhipeng, Zheng',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    install_requires=[
        'pandas',
        'numpy',
        'sklearn',
        'tables',
        'pyyaml',
        'click',
        'coloredlogs'
    ],
    entry_points={
        "console_scripts": ["chitaxi = chitaxi.cli:main"]
    }
)
