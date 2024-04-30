# Speech-Understanding-PA-03

if you are reading this readme then you are in the root directory of the code.

## Overview
This document provides step-by-step instructions for setting up the Speech Understanding PA-03 project. If you encounter any issues during setup, please refer to the troubleshooting section at the end of this document.

## Getting Started
1. Create a python environment using conda - `conda create --prefix ./.venv python=3.9`
2. activate the python environment - `conda activate ./.venv`
3. install requirements - `pip install -r requirements.txt`
4. install `fairseq` or copy it from the original source as mentioned in github link provided in assignment. if copying `fairseq` then place it inside the `code` directory. 
5. download `SSL model`, data and `xlsr2_300m.pt` file from the github link in paper.
6. copy `model.py` from the github repo provided in paper.
7. Edit `cp_path` in model.py to point downloaded `xlsr2_300m.pt` file.

## Running the Code
8. open `code\main.py` edit  all the inputs.
9. Run main.py from the root directory `python code\main.py` 

## Additional Notes

- For step 7, ensure you download the SSL model, data, and `xlsr2_300m.pt` file from the correct location mentioned in the GitHub link provided in the paper.

- Deactivate the environment after running the code using:

- If you encounter any issues, refer to the troubleshooting section below or consult the original source for `fairseq` installation.

## Troubleshooting

If you encounter any issues during setup or execution, consider the following tips:

- Check that all dependencies are installed correctly.
- Ensure that paths in `model.py` and `main.py` are correctly set.
- Refer to the original source for `fairseq` installation if you encounter issues with its setup.
- Reach out to the project maintainers for further assistance.