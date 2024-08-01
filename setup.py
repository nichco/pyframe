from setuptools import setup, find_packages

setup(
    name='pyframe',
    version='0.0.0',
    author='Nicholas Orndorff',
    author_email='norndorff@ucsd.edu',
    license='MIT',
    python_requires='>=3.7',
    platforms=['any'],
    install_requires=[
        'numpy',
        'pytest',
        'gitpython',
        'setuptools',
        'pyvista',
    ],
)
