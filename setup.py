from setuptools import setup, find_packages

def get_version() -> str:
    init = open("__init__.py", "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name="onpolicy",
    version=get_version(),
    description="on-policy algorithms of marlbenchmark",
    packages=find_packages(),
    python_requires='>=3.6',
)