# Visual-navigation
A Bachelor Thesis project on Navigation for Autonomous Vehicles

How to run:

1) Download the YOLO weights in tensorflow format from the following link:
https://drive.google.com/drive/folders/1m8tKgusgaaQt5GahSPuzWY4oQd27Lpg9?usp=sharing
Put these files in a folder named "saved_model"

2) Put a video in the current directory and rename it as "input_video.mp4" on which navigation is to be done.

3) Create a folder called "output" to store the output frames from the pipeline and intermediate ouputs.

4) Make sure you have the following dependencies installed in your python environment:
    i) tensorflow-gpu
    ii)opencv 4.0.0
    iii)opencv-python --------required for tracking
    iv)cuda-nn 9.0
    
5) Run the command "python vscript.py" in the terminal.
