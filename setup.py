import setuptools

setuptools.setup(
    name="genereg", # Replace with your own username
    version="0.1.0",
    author="Anton Afanassiev",
    author_email="anton.afanassiev@alumni.ubc.ca",
    description="A package to perform regression on gene expression data with OT fates.",
    url="https://github.com/Jabbath/regression_devel",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)