from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'

def get_requirements(path):
    with open(path) as f:
        requirements = f.read().splitlines()
    
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    
    return requirements
    


setup(
    name='ml project',
    version='0.0.1',
    author='Syed',
    author_email='srizwan232326@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)