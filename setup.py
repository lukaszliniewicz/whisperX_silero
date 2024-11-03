import os
import platform

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="whisperx_silero",
    py_modules=["whisperx"],
    version="3.1.1",
    description="Time-Accurate Automatic Speech Recognition using Whisper and Silero VAD.",
    readme="README.md",
    python_requires=">=3.8",
    author="Lukasz Liniewicz",
    url="https://github.com/lukaszliniewicz/whisperX_silero",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
    entry_points={
        "console_scripts": ["whisperx=whisperx.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
