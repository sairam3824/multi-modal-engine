from setuptools import setup, find_packages

setup(
    name="multimodal-rag-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.9",
    author="Your Name",
    description="RAG pipeline for multimodal documents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
)
