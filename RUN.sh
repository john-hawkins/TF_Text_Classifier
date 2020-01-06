#!/bin/bash

# FIRST BUILD THE MODEL AND TEST IT

python3 train_model.py

python3 test_model.py


# If this has all been successful we create the deployment artefacts 

# THE ENVIRONMENT
cd dr_deploy/env

########################################################
# -- OPTIONAL -- 
# Replace requirements.txt if your version of tensorflow is different
# pip freeze | grep tensorflow > requirements.txt
 
./CREATE_ENVIRONMENT.sh

# THEN THE MODEL ARCHIVE 
cd ../model

./CREATE_MODEL_ARCHIVE.sh

