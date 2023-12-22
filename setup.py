from setuptools import find_packages,setup


def get_requirements(filename):
    with open(filename,'r') as file_obj:
        requirements=[str(req.replace('\n','')) for req in file_obj.readlines() if req!='-e .']
        return requirements
setup(
    name='Mark Prediction',
    version='0.0.1',
    author='Abhimanyu M A',
    author_email='abhimanyuma75@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)