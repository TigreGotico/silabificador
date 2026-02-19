from setuptools import setup, find_packages

setup(
    name="silabificador",
    version="1.0.0",
    description="A syllabifier for Portuguese",
    author="JarbasAI",
    author_email="jarbasai@mailfence.com",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires=">=3.7",
)
