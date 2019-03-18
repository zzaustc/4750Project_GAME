#!/bin/bash

cp welcome.png view.png
sbatch --gres=gpu:1 --time=3 --wrap="python Game.py"
