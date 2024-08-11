
# AdobeGenSolve_AInsteinsProphecy
This repository is the submission of the solution for Adobe GenSolve AI Hackathon Round 2 by team AI-nstein's Prophecy.
What it does:

1. **Regularisation**: Beautifies and regularises hand-drawn curves and shapes into perfect ones.

2. **Symmetry Detection**: Detects the line of symmetry and prints out the equation of the line.

3. **Curve Completion**: Completes incomplete/ occluded curves and shapes by defining their line of symmetry.

After taking the input, the program plots the output .csv file using Matplotlib and stores the output file in the output folder.

#### Step 1
```
pip install -r requirements.txt
```

#### Step 2
Run the file by passing the task that needs to be done, for instance, `--task` (`regularisation` or `occlusion`) along with the csv file by using `--path` and path of the `.csv` file.

For example:
```
python main.py --task regularisation --path dataset\isolated.csv
```

#### Step 3
The resulting file is in `.csv` format stored at `output\outputfile.csv`.


Developed by:
- Suryansh Srivastava
- Piyush Ojha
- Umesh Chavda