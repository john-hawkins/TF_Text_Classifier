Tensorflow Text Classifier on DataRobot
---------------------------------------

In this project we take one of the TensorFlow 2.0 Examples and make it deployable
as a standalone model that runs on a DataRobot Prediction Server.

https://www.tensorflow.org/tutorials/text/text_classification_rnn

The model expects the input to be a single block of text in each record and it
will provide a sentiment score which represents the probability that the review
is positive.

## Instructions

Train the model using the script [train_model.py](train_model.py)

Then test it with the script [test_model.py](test_model.py)

Now go into the [dr_deploy](dr_deploy) directory and create the archives for both
* [environment](dr_deploy/env)
* [model](dr_deploy/model)

These two archives need to be uploaded into the DataRobot MLOps interface to create
the deployment.


