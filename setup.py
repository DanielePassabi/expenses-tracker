"""Setup Settings for Package."""

from setuptools import setup

if __name__ == '__main__':
    print('Building package.')

    requirements = [
        'numpy',
        'pandas',
        'plotly',
    ]

    setup(install_requires=requirements)
