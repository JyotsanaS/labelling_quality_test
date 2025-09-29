from setuptools import setup, find_packages
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        name="raga_models.executor",
        sources=["executor.py"],
    )
]

setup(
    name="raga_models",
    version="0.0.13.2.8",
    author="RagaAI",
    author_email="datascience@raga.ai",
    description="Generates mistake score",
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3"),
    install_requires=[
        "boto3==1.33.1",
        "botocore==1.33.1",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "tqdm==4.66.1",
        "numpy==1.24.3",
        "opencv_contrib_python==4.6.0.66",
        "opencv_python==4.6.0.66",
        "opencv_python_headless==4.8.1.78",
        "pandas==2.0.2",
        "segment_anything==1.0",
        "segment_anything_py==1.0",
        "tifffile==2023.8.30",
    ],
)
