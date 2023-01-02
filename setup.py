from typing import List

from setuptools import find_packages, setup

import src.maud


def get_requirements() -> List[str]:
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


setup(
    name="maud",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=get_requirements(),
)
