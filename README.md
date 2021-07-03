Tips to run this on your own PC:

Use git or Github Desktop to clone https://github.com/Aminnnnnnnnn/classifican

Install Python if you havn't already. This was tested using Python 3.8 on Linux

From your command line, change to the local directory this has been cloned to

cd ~/Documents/Github/classifican

Satisfy the dependencies using the "pip" command to install python libraries:

pip install tensorflow
pip install matplotlib
pip install opencv-python

Run the training program to re-create the data set on your PC:

python train.py

Run the summary program to check that you have a valid model:

python model_summary.py

Finally, run the predict program against the example images provided:

python modem_predict.py

Edit this file to point it to your own images, and try again!

