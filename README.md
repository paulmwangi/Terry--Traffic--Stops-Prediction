# Terry Traffic Stops: Predicting Arrest Outcomes
Author: [Paul Mwangi](https://github.com/paulmwangi)


## Overview
In the pivotal 1967-8 Supreme Court decision of Terry v. Ohio, it was determined that a police officer’s conduct of stopping and searching individuals based solely on suspicious behavior did not breach the Fourth Amendment’s protection against “unreasonable search and seizure.” This ruling gave rise to the concept of “reasonable suspicion,” a legal standard allowing police to briefly detain individuals without the more substantial proof necessary for formal arrests. “Terry Stops” refer to the practice of pulling over drivers under this suspicion.

## Business Understanding
### Problem Statement
Law enforcement agencies need insights into the factors influencing arrest outcomes during Terry Stops. By predicting whether an arrest will occur, we can enhance decision-making and potentially reduce unnecessary detentions.

### Specific Objectives
1. Develop a robust binary classification model.
2. Identify key features that contribute to arrest decisions.
3. Investigate the role of gender and race in arrest outcomes.

### Research Questions
1. What factors are most influential in determining whether an arrest occurs after a Terry Stop?
2. How does the presence of weapons impact arrest outcomes?
3. Is there evidence of bias related to gender or race?

### Success Criteria
- Achieve an accuracy of at least 80% in predicting arrest outcomes.
- Provide actionable insights for law enforcement agencies.

## Data Understanding
### Data Description

Our dataset is sourced from the.[Seattle Police Department's Terry Stops](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8/about_data)
This dataset includes the following features:
- **Arrest Outcome**: Binary (1 for arrest made, 0 for no arrest).
- **Presence of Weapons**: Whether weapons were found during the stop.
- **Time of Day**: Categorized into morning, afternoon, evening, and night.
- **Other Relevant Features**: Additional contextual information.

## Data Preparation
### Data Selection
We will focus on relevant features related to the stop outcome, excluding any personally identifiable information (PII).

### Data Cleaning
1. Handle missing values.
2. Encode categorical variables.
3. Address class imbalance (if present).

## Modeling
We will explore various classification algorithms, including logistic regression, decision trees, and random forests. Hyperparameter tuning will be performed to optimize model performance.

## Evaluation
We will assess the model using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Conclusion
Our project aims to provide actionable insights for law enforcement agencies, improve decision-making during Terry Stops, and address potential bias. Transparency and ethical considerations will guide our analysis.

## Recommendations
1. Consider implementing model predictions in real-time during Terry Stops.
2. Continuously monitor and evaluate the model's performance.
3. Promote transparency by sharing findings with stakeholders and the public.

## Repository Structure

* ├── .ipynb_checkpoints
* ├── images
* ├── .gitignore
* ├── README.md
* ├── Terry-Stops-Classification
* ├── Terry-Stops.csv
* └── presentation.pdf
