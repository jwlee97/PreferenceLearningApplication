# PreferenceLearningApplication

## Preamble
This toolkit was developed for the dissertation, 'A model-based design tool for 3D GUI layout design that accommodates user attributes' submitted in 2021 for the 
MPhil in Machine Learning & Machine Intelligence program at the University of Cambridge.

## Instructions
In test.py, change the image file, panel sizes, and objective function weights in main() to adjust the UI layout.
At each iteration, choose the preferred UI layout - this will become the 'preference' image for the next iteration, and the application will suggest a new layout as the 'suggestion'.
The 'preference' image for the first iteration will be the output of weighted sum optimization in UIoptimization.py using the function weights specified.
When finished, press 'quit'. The script will display the optimized UI layout when finished, and output the optimal locations of each panel in world coordinates.

<p align="center">
  <img src="https://github.com/jwlee97/PreferenceLearningApplication/blob/master/preference_learning.jpg" width=800 />
</p>
