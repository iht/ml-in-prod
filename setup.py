"""SetupTools main script."""
import os
import versioneer
from setuptools import find_packages, setup

readme = os.path.join(os.path.dirname(__file__), 'README.md')

setup(name='my_first_ml_model',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description="My first ML model in production",
      long_description=open(readme).read(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: Other/Proprietary License',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development :: Build Tools',
      ],
      keywords='Some keyword',
      author='Some author',
      author_email='some@email.com',
      url='Some URL',
      license='All rights reserved',
      packages=find_packages(exclude=['tests']),
      include_package_data=True,
      zip_safe=False
      )
