"""SetupTools main script."""
#   Copyright 2022 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import versioneer
from setuptools import find_packages, setup

readme = os.path.join(os.path.dirname(__file__), 'README.md')

with open(os.path.join(os.path.dirname(__file__),
                       'requirements.txt')) as requirements:
    install_requires = requirements.readlines()

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
      install_requires=install_requires,
      include_package_data=True,
      zip_safe=False
      )
