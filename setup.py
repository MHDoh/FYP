from setuptools import setup, find_packages

setup(
    name="eyes",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'ultralytics>=8.0.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'PyQt5>=5.15.0'
    ],
)