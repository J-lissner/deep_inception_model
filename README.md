# Deep Inception Neural Network
Code to train a deep inception model, as well as an accompanying post processing/evaluation script. 
The parameters of a sample model are given with this repository. The post processing script showcases how to load the model.

Note that this code relies heavily on the `python scripts` repository  (https://github.com/J-lissner/python_scripts). Thus, it is recommended to clone this repository as well, and add the scripts into the `$PYTHONPATH`.

**Note:** The python code has been developed in `tensorflow version 2.11.0`. Latest changes in tensorflow (seen for 2.16) have broken the evaluation of the deep inception module and a rollback might be neccessary. 
