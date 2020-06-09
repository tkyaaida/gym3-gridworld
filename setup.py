from setuptools import setup, find_packages


setup(
    name='gym3-gridworld',
    version='0.0.1',
    description='Customizable 2D gridworld with gym3 interface',
    license='MIT',
    author='Takuya Aida',
    author_email='tkyaaida@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gym3==0.3.1', 'torch==1.5.0'
    ],
    extras_require={
        'dev': ['pytest', 'sphinx', 'line_profiler']
    }
)
