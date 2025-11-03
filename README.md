# CD_ML_Denoising
Code for the training and deployment of denoising in CLAS12's Central Detector Tracking system.

## Requirements

All required libraries are detailed in env_install.txt. To install using venvs and pip in a new environment called newenvname (you should choose a better name):

      python3 -m venv /path/to/env/location/newenvname
      source /path/to/env/location/newenvname/bin/activate.csh
      pip install torch torchvision torchaudio
      pip install matplotlib
      pip install hipopy
      pip install scikit-learn
      pip install lightning
      pip list

Remember to always activate your environment before running code with source /path/to/env/location/newenvname/bin/activate.csh .

A script is provided to launch an interactive gpu session on ifarm. This is run with:

      source gpu_interactive
      module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
      module load cuda/12.4.1
      
Note that in that case, we need to create a GPU friendly PyTorch environment. We can replace the earlier line with:

      module load cuda/12.4.1
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
      pip install matplotlib
      pip install hipopy
      pip install scikit-learn
      pip install lightning
      pip list

## To Produce Training Data

The HipoParser class is used to produce training and testing datasets. These utilise a special bank CVT::MLHit. The data will be reading in PyTorch files .pt by running:

      python3 saveAndPlot.py

This separates the data into three sectors using the following mapping:

      sector_mapping = {
          1: {  # File 1
              (1, 2): [1, 2, 3, 4],
              (3, 4): [1, 2, 3, 4, 5, 6],
              (5, 6): [1, 2, 3, 4, 5, 6, 7],
              (7, 12): [1],
          },
          2: {  # File 2
              (1, 2): [4, 5, 6, 7, 8],
              (3, 4): [6, 7, 8, 9, 10],
              (5, 6): [7, 8, 9, 10, 11, 12, 13],
              (7, 12): [2],
          },
          3: {  # File 3
              (1, 2): [8, 9, 10],
              (3, 4): [10, 11, 12, 13, 14],
              (5, 6): [14, 15, 16, 17, 18],
              (7, 12): [3],
          },
      }

The data is further scaled between 0 and 1 using the min/max definitions at the top of the saveAndPlot.py script, separated into training and testing scripts and padded to have a maximum number of hits in an event as 450. This padding is necessarry for the Java inference. The script will make plots of the input data before and after scaling, saved by default in the plots/inputs/ and plots/scaledVars/ directories. The Plotter class contains the plotting function definitions. Variables at the top of the script are used to define the location where plots are written (eg plots/) and an "end_name" string to append to plot names so that they are not overwritten.

Note that due to the separation per sector, it may be that events contain all noise. So far we chose to keep this in the training data so that the network will learn to predict only background.

## To Train

In the new environment do:

      python3 train.py

This will launch a training script that uses the base model definition in Classifier.py. The model is defined in the LitConvAutoencoder class which uses pytorch and pytorch-lightning for the model definition. 

The model is based on a GravNet architecture (see https://arxiv.org/abs/1902.07987) which learns a latent space representation (of size s_dim) of the data and creates dynamic graph representation of the data by clustering in this space. Note that in the default version, s_dim is set to one. A custom implementation of the GravNet code is used here due to the fact that the Java inference code can only read native PyTorch functionality and not PyTorch Geometric.

The trained network will be saved in the nets/ directory. The train script also produces some plots, saved by default in the plots/training/ directory. The test.py script can be used to make the plots without retraining. The Plotter class contains the plotting function definitions. Variables at the top of both scripts are used to define the location where plots are written (eg plots/), an "end_name" string to append to plot names so that they are not overwritten, and the sector number.


## To Deploy

The repository includes a toy maven project that allows to load the trained networks using the Deep Java Libray and applies them to a toy input. The pom.xml file contains the required set-up (assumes jdk/17) and the src/main/java/org/example/Main.java class contains a class capable of loading and applying the model. The class contains two main parts, first a translator that allows to convert the data into an input suitable for the network and convert the output of the network into a 2D array of floats. Second, the a Criteria class defines the path to the model and the engine (in this case pytorch).

      module load jdk/17.0.2
      mvn install -U
      mvn exec:java -Dexec.mainClass="org.example.Main"

The model output is compared to that obtained when training and testing the model. Note, you can expect some small differences when setting the GravNet s_dim to larger than 1 (it is set to 1 by default). 
