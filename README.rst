========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |requires|
    * - package
      - |version| |downloads| |wheel| |supported-versions| |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/Praetorius_Goldberg_2016/badge/?style=flat
    :target: https://readthedocs.org/projects/Praetorius_Goldberg_2016
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/eli-s-goldberg/Praetorius_Goldberg_2016.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/eli-s-goldberg/Praetorius_Goldberg_2016

.. |requires| image:: https://requires.io/github/eli-s-goldberg/Praetorius_Goldberg_2016/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/eli-s-goldberg/Praetorius_Goldberg_2016/requirements/?branch=master

.. |version| image:: https://img.shields.io/pypi/v/nanogbcdt.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/nanogbcdt

.. |downloads| image:: https://img.shields.io/pypi/dm/nanogbcdt.svg?style=flat
    :alt: PyPI Package monthly downloads
    :target: https://pypi.python.org/pypi/nanogbcdt

.. |wheel| image:: https://img.shields.io/pypi/wheel/nanogbcdt.svg?style=flat
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/nanogbcdt

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/nanogbcdt.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/nanogbcdt

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/nanogbcdt.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/nanogbcdt


.. end-badges

Generated with https://github.com/ionelmc/cookiecutter-pylibrary

Installation
============

::

    pip install nanogbcdt

Documentation
=============

Please find our documentation at https://github.com/eli-s-goldberg/nanogbcdt/wiki. Documentation will also
be made available at https://Praetorius_Goldberg_2016.readthedocs.io/, as soon as possible.

This work is to support the paper, "Discrimination of natural and engineered nanoparticles based on their multi-element fingerprints", accepted to Environmental Science: Nano in November, 2016.

Dr. Antonia Praetorius is the lead author. Model development was performed and is maintained by Dr. Eli Goldberg. Experimental work and other significant contributions are provided from the following authors:

Antonia Praetorius:superscript:`1,2`, Alexander Gundlach-Graham:superscript:`3`, Eli Goldberg:superscript:`4`, Willi Fabienke:superscript:`1`, Jana Navratilova:superscript:`1`, Andreas Gondikas:superscript:`1`, Ralf Kaegi:superscript:`5`, Detlef Günther:superscript:`3`, Thilo Hofmann:superscript:`1,2*`,  Frank von der Kammer:superscript:`1*`.

Affiliations are as follows:
* :superscript:`1` University of Vienna, Department of Environmental Geosciences and Environmental Science Research Network, Althanstr. 14, UZA II, 1090 Vienna, Austria
* :superscript:`2` University of Vienna, Research Platform Nano-Norms-Nature, Vienna, Austria
* :superscript:`3` ETH Zurich, Laboratory of Inorganic Chemistry, Vladimir-Prelog-Weg 1, 8093 Zurich, Switzerland
* :superscript:`4` ETH Zurich, Institute for Chemical and Bioengineering, Vladimir-Prelog-Weg 1, 8093 Zurich, Switzerland
* :superscript:`5` Eawag, Swiss Federal Institute of Aquatic Science and Technology, Überlandstr. 133, 8609 Dübendorf, Switzerland

Corresponding author emails are: thilo.hofmann@univie.ac.at, frank.kammer@univie.ac.at.

For model-related questions, feel free to email Eli Goldberg at elisgoldberg@gmail.com.


Examples
===========

Now that you have python3 and the dependencies installed, let's get you up and running on the software. To do this, I've included a notebook that outlines how to use the API by example in `Examples/distinguish_nat_v_tech_API.ipynb <https://github.com/eli-s-goldberg/Praetorius_Goldberg_2016/blob/master/Examples/distinguish_nat_v_tech_API.ipynb>`_) file. This is a special file that is able to be uploaded and presented in evaluated form on Github. It is also available for you to use, if you know how to use Jupyter notebook.

Using the provided databases
----------------------------

We've provided the example databases that go along with the work. These are contained in the `Databases` folder. Within this folder, we've included our training data (see API documentation for how to clean this data properly), as well as the test data. We will generate reports that evaluate the accuracy of the classifier within the training set, as well as the results of applying the trained classifier to the test data. These are available in the recently accepted publication.
Also in the examples folder are the training data and a correlation plot of the training data.


Development
===========

Obtaining the script
----------------------------

I do have aspirations of making this a deployable package, but for now I think it's best to keep it simple. The easiest method would be to download the file directory using the button above. This will download the all the files into a seperate folder. It should be able to be run without problem from this folder.

The preferred method is to use git to fork the directory to your github account so you can continue development. To do this, press the 'fork' button. Once you've forked the model to your personal github account, use git to download the model.

Once forked, find the clone github address, copy it, and paste it into a terminal prompt. This will copy the contents of the github page into a directory and initalize it for git. Please navigate to a convenient place before cloning the directory. I've included what this would look like from this directory:
::

git clone https://github.com/eli-s-goldberg/nanogbcdt.git



To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
