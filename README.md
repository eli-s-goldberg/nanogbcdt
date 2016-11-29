# Distinguishing engineered nanomaterials from natural nanomaterials using gradient-boosted classification decision trees

This work is to support the paper, "Discrimination of natural and engineered nanoparticles based on their multi-element fingerprints", accepted to Environmental Science: Nano in November, 2016.

Dr. Antonia Praetorius is the lead author. Model development was performed and is maintained by Dr. Eli Goldberg. Experimental work and other significant contributions are provided from the following authors:

Antonia Praetorius<sup>1,2</sup>, Alexander Gundlach-Graham^3^, Eli Goldberg^4^, Willi Fabienke^1^, Jana Navratilova^1^, Andreas Gondikas^1^, Ralf Kaegi^5^, Detlef Günther^3^, Thilo Hofmann^1,2*^,  Frank von der Kammer^1*^.

Affiliatios are as follows:
^1^University of Vienna, Department of Environmental Geosciences and Environmental Science Research Network, Althanstr. 14, UZA II, 1090 Vienna, Austria
^2^University of Vienna, Research Platform Nano-Norms-Nature, Vienna, Austria
^3^ETH Zurich, Laboratory of Inorganic Chemistry, Vladimir-Prelog-Weg 1, 8093 Zurich, Switzerland
^4^ETH Zurich, Institute for Chemical and Bioengineering, Vladimir-Prelog-Weg 1, 8093 Zurich, Switzerland
^5^Eawag, Swiss Federal Institute of Aquatic Science and Technology, Überlandstr. 133, 8609 Dübendorf, Switzerland

Corresponding author emails are: thilo.hofmann@univie.ac.at, frank.kammer@univie.ac.at.

For technical work or questions, feel free to email Eli Goldberg at <elisgoldberg@gmail.com>.


## Installation instructions

This installation is for Mac OSX. Instructions are not given for a virutal environment, but it should work.


### Obtaining python3
The model is programmed in python3. If you don't have 3, please follow the following instructions.

1. Install homebrew.
    Paste the following into a terminal prompt.

    `'/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

    Update homebrew by pasting the following into a terminal prompt.

    `brew update`

2. Use homebrew to install python3.
    Paste the following into a terminal prompt:

    `sudo brew install python3`

### Installing git
Git is a great way to make code development easier. It's powerful, but the basics are super simple. To make things easy, use brew to install git.

`sudo brew install git`

### Obtaining the script
I do have aspirations of making this a deployable package, but for now I think it's best to keep it simple. The easiest method would be to download the file directory using the button above. This will download the all the files into a seperate folder. It should be able to be run without problem from this folder.

The preferred method is to use git to fork the directory to your github account so you can continue development. To do this, press the 'fork' button. Once you've forked the model to your personal github account, use git to download the model.

Once forked, find the clone github address, copy it, and paste it into a terminal prompt. This will copy the contents of the github page into a directory and initalize it for git. Please navigate to a convenient place before cloning the directory. I've included what this would look like from this directory:

`git clone https://github.com/eli-s-goldberg/Praetorius_Goldberg_2016.git`

### Installing the dependencies

#### Installing pip
Pip is another package manager, but specifically for python packages. If pip is already installed, there's no need to go through this process again. To determine if pip is already installed, paste this into a terminal prompt:

`which pip`

If you come back with something like: `/usr/local/bin/pip`, then you're all good. If you don't, paste the following into a terminal prompt to install the "easy_install" package manager:

`curl https://bootstrap.pypa.io/ez_setup.py -o - | sudo python`

Once this is complete, use easy_install to install pip by pasting the following into a terminal prompt:

`sudo easy_install pip`

#### Using pip to install the package dependencies

Once you have all the package managers (I know, it's a bit much), follow these instructions to make sure you have all the packages required to run the model. The easiest way to do this is to use pip to install the requirements. The package requirements are contained in the 'requirements.txt' file. Navigate to the directory that contains this file and past in the following into a terminal prompt:

`sudo pip3 install -r requirements.txt`

This should install the packages, and their sub-dependencies.

## Running the software - See Examples folder

Now that you have python3 and the dependencies installed, let's get you up and running on the software. To do this, I've included a notebook that outlines how to use the API by example in [`Examples/distinguish_nat_v_tech_API.ipynb`](https://github.com/eli-s-goldberg/Praetorius_Goldberg_2016/blob/master/Examples/distinguish_nat_v_tech_API.ipynb) file. This is a special file that is able to be uploaded and presented in evaluated form on Github. It is also available for you to use, if you know how to use Jupyter notebook.

### Using the provided databases.
We've provided the example databases that go along with the work. These are contained in the `Databases` folder. Within this folder, we've included our training data (see API documentation for how to clean this data properly), as well as the test data. We will generate reports that evaluate the accuracy of the classifier within the training set, as well as the results of applying the trained classifier to the test data. These are available in the recently accepted publication.
Also in the examples folder are the training data and a correlation plot of the training data.