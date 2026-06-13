# DonorsChoose Funding Risk: ML for Targeted Intervention

Machine learning model that flags the DonorsChoose classroom funding requests most at risk of going unfunded, so limited reviewer attention reaches under-resourced schools first. Includes a fairness audit of model behavior across school poverty levels.

## The policy problem

Teachers across the United States routinely finance classroom supplies out of pocket or turn to crowdfunding platforms like DonorsChoose. About one-third of posted projects never reach their funding goal, and failure falls hardest on students in under-resourced schools. DonorsChoose has limited capacity for human review and personalized intervention, so the question this project answers is operational: which 10% of newly posted projects should a reviewer look at first.

## What the model does

- Predicts the probability that a newly posted project will be fully funded, using only features available at posting time (school characteristics, project category, requested amount).
- Compares several classifiers under 5-fold stratified cross-validation; XGBoost performed best with a mean cross-validation ROC AUC of 0.766.
- Tunes the XGBoost pipeline with grid search and selects a decision threshold using Youden's J statistic, balancing the cost of missed at-risk projects against reviewer capacity.
- On held-out test data of 185,103 projects, the final model reaches a ROC AUC of 0.757. At the selected threshold of 0.68 it correctly identifies 68% of unfunded projects while retaining 69% recall on funded ones.
- Ranks projects by predicted funding probability and surfaces the bottom 10% as the recommended intervention list.

An honest note on tuning: grid search did not beat the default XGBoost configuration on cross-validation score (best tuned ROC AUC 0.756 vs 0.766 for the default). The final model is reported on the held-out test set either way, and the gap is documented in the notebook rather than hidden.

## Fairness audit

Because the model would steer real intervention resources, the final notebook audits error rates across sensitive groupings, including school poverty level and primary focus area. The audit reports per-group accuracy, precision, recall, specificity, false positive rate, and false negative rate, so that a deployment decision can weigh whether the model treats high-poverty schools differently from low-poverty ones.

## Repository map

| File | Purpose |
|---|---|
| `ML_Final.ipynb` | Main end-to-end notebook: cleaning, model comparison, tuning, threshold selection, at-risk project list, bias audit |
| `ML-Project.pdf` | Full 18-page written report with executive summary, methodology, and policy recommendations |
| `eda.ipynb` | Exploratory data analysis of the raw DonorsChoose tables |
| `archive/` | Development-history notebooks: early pipeline prototypes, cross-validation and grid-search experiments, feature engineering |

## Data

The data comes from the KDD Cup 2014 DonorsChoose competition on Kaggle (projects, outcomes, donations, essays, and resources tables). The files are too large to commit, so download them from Kaggle and place them in a `Datasets/` folder at the repo root before running the notebooks.

Dataset link: https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Download the Kaggle data into `Datasets/` as described above.
3. Open `ML_Final.ipynb` and run top to bottom. Expect the cross-validation and grid search sections to take a while on the full dataset.

## Limitations and future work

- The model uses only static, at-posting features. It does not estimate the effect of any specific intervention, so it identifies where to intervene but not how.
- A production version would pair the model with a database-backed decision support system and a reviewer-facing interface for ad hoc analysis.
- Intervention analysis, including measuring uplift from reviewer contact, is the natural next step.

## Team

Built as a team project for the Machine Learning Foundations course at Carnegie Mellon University (Spring 2025) by Iqbal, Savellano, and Zavala Reina.

## License

Released under the MIT License. See [LICENSE](LICENSE).
