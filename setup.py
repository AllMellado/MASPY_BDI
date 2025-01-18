from setuptools import setup, find_packages

VERSION = '0.5.4'
DESCRIPTION = 'Multi-Agent System for Python (MASPY) with Machine Learning proprieties'
LONG_DESCRIPTION = 'A library for the devolopment of multi-agent systems with components of machine learning'

# Setting up
setup(
    name="maspy-ml",
    version=VERSION,
    author="Alexandre Mellado",
    author_email="<melladoallm@gamil.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    package_data={"maspy": ["py.typed","*.pyi"]},
    install_requires=['numpy','pandas','openpyxl'],
    keywords=['python', 'autonomous agents', 'multi-agent system', 'machine learning'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
    ]
)