from setuptools import setup, find_packages

setup(
    name="ai-voice-transfer",
    version="0.1.0",
    description="Make any LLM write like a specific human, not like an AI.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="David Dabert",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Linguistic",
    ],
)
