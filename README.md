# 4750Project_GAME
A Simple Parallel Computer Game Using PyOpenCL

#E4750 PROJECT: GAME
author Zian Zhao (zz2558)
##Overview
This program is prepared for EECSE4750_001_2018_3 - HETEROGEN COMP-SIG PROCESSING. The code can only be run on Tesseract server!

##Description & How to run
####City.py

Code for 3D modeling, build a city and save it as .npy format.

To run the code:
> $ python City.py

--
####Projection.py

Core parallel algorithm, providing functions which takes a 3D model and returns a 2D RGB projection image with runtime of kernel code.

This model will be imported by Game.py and Runtime.py

--
####Game.py

Code for playing the game. Some parameters are changable.

To run the code:

Before run the code, make sure you have already run City.py, and there are MyCityModel0.npy or MyCityModel1.npy in the folder. Then open another terminal, cd /source

*It's a little bit tricky! you need to open a new terminal.
> $ xdg-open view.png

A window of image will be displayed on the screen. Then run the code. I have a script 'play.sh' prepared for it.
> $ sh play.sh

If 'xdg-open' does not work, download view.png after the code runs. Have a look at the last view.

--
####Runtime.py

This code evaluates the runtime of naive version and optimized version. It generates a plot 'projection_runtime.png'.

To run the code:

> $ sbatch --gres=gpu:1 --time=3 --wrap="python Runtime.py"
