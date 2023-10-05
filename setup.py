from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="hottakes",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": [
            # List any development dependencies here
        ]
    },
)
