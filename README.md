Tensorflow Text Classifier on DataRobot
---------------------------------------

In this project we take one of the TensorFlow 2.0 Examples and make it deployable
as a standalone model that runs on a DataRobot Prediction Server.

https://www.tensorflow.org/tutorials/text/text_classification_rnn

The model expects the input to contain a column called 'review' which contains
a single block of text. The model will return a sentiment score which represents 
the probability that the review is positive.

## Instructions

Train the model using the script [train_model.py](train_model.py)

Optional: You can then test it with the script [test_model.py](test_model.py)

Once the model is built it can be deployed into DataRobot MLOps

To do this you will need to create a suitable environment and model archive.

Go into the [dr_deploy](dr_deploy) directory and create the archives for both
* [environment](dr_deploy/env)
* [model](dr_deploy/model)

These two archives need to be uploaded into the DataRobot MLOps interface to create
the deployment.

You can also use the [RUN Script](RUN.sh) to execute all of these commands.

```
./RUN.sh > log.txt
```


