import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='kgformula',
     version='0.1',
     author="Robert Hu",
     author_email="robert.hu@stats.ox.ac.uk",
     description="KC-HSIC: Kernel Causal HSIC",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/MrHuff/KCHSIC",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )