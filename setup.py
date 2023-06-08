from setuptools import setup

setup(
    name="meekovina",
    version="0.3.0",
    install_requires=[
        "pyyaml", "numpy", "scipy", "pandas", "rdkit", "meeko>=0.4.0",
    ],
    entry_points={
        'console_scripts': [
            'meekovina=meekovina.meekovina:main',
        ],
    },
    author="Michio Katouda",
    author_email="katouda@rist.or.jp",
    description="Python script easy to use Autodock-Vina basic docking simualation",
    url="https://github.com/mkatouda/meekovina",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.7',
)
