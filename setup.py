from setuptools import setup, find_packages

with open('README.rst', 'r') as f:
    readme = f.read()

with open('LICENSE', 'r') as f:
    license = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read()

setup(
    name='mlmodel',
    version='0.1.2',
    description='Sturdy interface for predictive ML models',
    long_description=readme,
    author='Samuel Prevost',
    author_email='samuel.prevost@pm.me',
    url='https://github.com/sam1902/MLModel',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=requirements
)

