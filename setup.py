from setuptools import setup, find_packages

setup(
    name="rholearn",
    version="0.0.0",
    packages=find_packages(
        include=[
            "rholearn", 
            "rholearn.*",
            "rholearn/aims_interface",
            "rholearn/aims_interface.*",
            "rholearn/utils",
            "rholearn/utils.*",
            ]
        ),
)
