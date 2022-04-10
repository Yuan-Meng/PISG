# code and data for "People Reproduce Disparities in Selective Sampling by Assuming the Sampler as A Knowledgeable Utility Maximizer"

This repo contains all data (anonymized) and code used to support the findings in Meng and Xu (submitted).

## data

To protect participant identity, we provide anonymized responses extracted from Qualtrics data, which is all you need to reproduce results in this study. The `data` folder has 3 subfolders that contain the following information:
1. `ready`: This folder contains all processed data in the form of 4 CSV files (Experiment 1: "6aliens"; Experiments 2 to 4: "20aliens"). Files with "behavior" in filenames are for reporting demographic information and analyzing how participants rated the agent. Files with "modeling" in filenames are used for computational modeling.
2. `designs`: This folder contains all trials contents used in Experiment 1 (subfolder: `6aliens_16trials`) and Experiments 2 to 4 (subfolder: `20aliens_24trials`). In `20aliens_24trials`, you can use the Jupyter notebook `find_best_trials.ipynb` to rank 230 possible trial contents and select the 24 that we used in our study.
5. `predictions`: This folder contains group hit rates inferred by 3 models and human participants in all experiments.

## modeling
This folder contains the Jupyter notebook (`infer_hit_rates.ipynb`) for inferring group hit rates. The Python file `pisg.py` has custom functions needed for modeling, which is imported as a module in this notebook.

## stimuli
This folder contains all stimuli used in this study.
1. `slides`: Stimuli in all conditions are originally in the form of Apple Keynote slides. Slides with `outcomes_unknown` in filenames don't show whether scanned aliens stole gems or not. Slides with `revealed` in filenames show this information. The number right before `inconsistent` indicates how many inconsistent trials there are in the given condition.
2. `video_intros`: This folder contains introductory videos used in the study. The `demo` videos introduce the rule of the Gem Patrol game whereas the `player` videos introduce the best border patrol officer Alex.
3. `materials`: The Jupyter notebook `group_generator_20aliens.ipynb` was used to progmatically generate 20 aliens in Experiments 2 to 4.

## results_for_paper
This folder contains coded used to outpout figures used in our manuscript. 
