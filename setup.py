from setuptools import setup, find_packages

setup(
    name="silabificador",
    version="0.1.0",
    description="A regional syllabifier for Portuguese using Brill taggers",
    author="JarbasAI",
    author_email="jarbasai@mailfence.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "joblib",
        "nltk"
    ],
    package_data={
        "silabificador": ["models/*.pkl"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires=">=3.7",
)
