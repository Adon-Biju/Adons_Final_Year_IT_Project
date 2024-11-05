# Adons_Final_Year_IT_Project

------Environment Setup (Installation done on MacOS)----
* Install Python 3.9
* Go to https://github.com/conda-forge/miniforge#miniforge3 and copy the following command to install Miniforge3 (the minimal installer for Conda specific to conda-forge)
* Press Enter to Continue
* Type in "yes" to accept license terms and then press Enter
* Verify the location as to where Miniforge will be installed in your directory and then once you are satisfied press Enter
* Wait for installation to occur
* Once installation has finished verify the conda installation by checking with the command "conda --version" and it should give you the conda version you have installed.
* Create a conda environment "conda create -n face_recognition_env python=3.9"
* Activate the new conda environment "conda activate face_recognition_env"
* Install face_recognition packages within this conda environment since some of the packages (specifically the dlib package within face_recognise cannot be installed outside this environment). Use this command -> "conda install -c conda-forge face_recognition"
* pip install opencv-python 
* Run python3 main.py



