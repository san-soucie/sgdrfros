import os
from glob import glob

from setuptools import find_packages, setup

package_name = "sgdrf"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (os.path.join("share", package_name), ["package.xml"]),
        (
            os.path.join("share", "ament_index", "resource_index", "packages"),
            [os.path.join("resources", package_name)],
        ),
        (os.path.join("share", package_name), glob("launch/*launch.[pxy][yma]*")),
    ],
    install_requires=["setuptools", "pyro-ppl", "torch"],
    zip_safe=True,
    maintainer="John San Soucie",
    maintainer_email="jsansoucie@whoi.edu",
    description="ROS 2 interface for SGDRFs",
    license="Apache 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["sgdrf_node = sgdrf.__main__:main"],
    },
)
