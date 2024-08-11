
# AdobeGenSolve_AInsteinsProphecy
This repository is the submission of the solution for Adobe GenSolve AI Hackathon Round 2 by team AI-nstein's Prophecy.
What it does:

1. **Regularisation**: Beautifies and regularises hand-drawn curves and shapes into perfect ones.

2. **Symmetry Detection**: Detects the line of symmetry and prints out the equation of the line.

3. **Curve Completion**: Completes incomplete/ occluded curves and shapes by defining their line of symmetry.

After taking the input, the program plots the output .csv file using Matplotlib and stores the output file in the output folder.

**Since sufficient data was not provided, the code runs on threshold values to identify shapes and regularise them. This threshold value may be modified for different test cases to obtain accurate results.**

SCOPE: This threshold value calculation may be automated using machine learning in the future.

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
```
python main.py --task fragmented --path dataset\frag1.csv
```
```
python main.py --task occlusion --path dataset\occlusion1.csv
```

**Note**: For regularisation problems that contain fragmented polylines, please use `--task fragmented`

#### Step 3
The resulting file is in `.csv` format stored at `output\outputfile.csv`.

Running Testcases:
---
Isolated Regularisation

<img src="https://github.com/user-attachments/assets/383e7ffb-60c8-42ed-bd02-3bc31e8e1890" alt="drawing" width="300"/>
<img src="https://github.com/user-attachments/assets/0031bbf3-0dc1-49a7-a87a-f82a5e08596e" alt="drawing" width="300"/>

---
Fragmented Regularisation

<img src="https://github.com/user-attachments/assets/8c18771a-6b7d-4695-bd65-1b0b9efc3b7b" alt="drawing" width="300"/>
<img src="https://github.com/user-attachments/assets/5e1e8c6e-99f5-42c0-8949-e5960f16cfe3" alt="drawing" width="300"/>

---
Occluded Shape Completion

<img src="https://github.com/user-attachments/assets/1111c4cc-5486-4351-977f-68e20bdfe94a" alt="drawing" width="300"/>
<img src="https://github.com/user-attachments/assets/3ab78bb6-83b3-4db5-8329-ababb4b4d5e6" alt="drawing" width="300"/>


---

Developed by Team AInstein's Prophecy:
- Suryansh Srivastava
- Piyush Ojha
- Umesh Chavda


