# from distutils.core import setup
from setuptools import setup, find_packages

# Define requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

description = """
PyTradeX is a propietary crypto currency forecasting library that is leveraged to run a trading bot 
that operates on Binance, trading cryptocurrency futures while following an ML based trading strategy.
"""

# Define setup
setup(
    name="PyTradeX",
    description=description,
    author="Simón P. García Morillo",
    author_email="simongmorillo1@gmail.com",
    version="1.0.0",
    install_requires=requirements,
    packages=find_packages(),
    package_data={"test": ["test/*"]}, # "config": ["config/*"], 
    long_description=open("README.md").read(),
    license=open("LICENSE").read()
)