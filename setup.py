from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mlmodel',
    version='0.1.0',
    description='Sturdy interface for predictive ML models',
    long_description=readme,
    author='Samuel Prevost',
    author_email='samuel.prevost@pm.me',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

