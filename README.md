# DonorsChoose Funding Risk: ML for Targeted Intervention

Machine learning model that flags the DonorsChoose classroom funding requests most at risk of going unfunded, so limited reviewer attention reaches under-resourced schools first. Includes a fairness audit of model behavior across school poverty levels.

**Fairness audit findings:** at the shared 0.68 decision threshold, error rates differ across school poverty levels. The model misses truly at-risk (unfunded) projects most often at the highest-poverty schools — 37.7% of their unfunded projects are wrongly predicted to be funded, versus 22.0–27.2% for the moderate-, low-, and high-poverty groups — while it flags projects that would have been funded anyway most often at low- and moderate-poverty schools (48.4% and 46.7%, versus 38.3% at high- and 24.3% at highest-poverty schools). The single global threshold balances overall error rates without equalizing them across groups; the notebook documents these gaps (group sizes run from 4,612 low-poverty to 107,283 highest-poverty test projects) rather than correcting them, so a deployment decision must explicitly choose whether to accept the disparity or move to per-group operating points.

## The policy problem

Teachers across the United States routinely finance classroom supplies out of pocket or turn to crowdfunding platforms like DonorsChoose. About one-third of posted projects never reach their funding goal, and failure falls hardest on students in under-resourced schools. DonorsChoose has limited capacity for human review and personalized intervention, so the question this project answers is operational: which 10% of newly posted projects should a reviewer look at first.

## Model design and performance

- Predicts the probability that a newly posted project will be fully funded, using only features available at posting time (school characteristics, project category, requested amount).
- Compares several classifiers under 5-fold stratified cross-validation; XGBoost performed best with a mean cross-validation ROC AUC of 0.766.
- Tunes the XGBoost pipeline with grid search and selects a decision threshold using Youden's J statistic, balancing the cost of missed at-risk projects against reviewer capacity.
- On held-out test data of 185,103 projects, the final model reaches a ROC AUC of 0.757. At the selected threshold of 0.68 (chosen by Youden's J to balance the two error rates) it correctly identifies 68% of unfunded projects (specificity 0.68) and recovers 69% of funded ones (recall 0.69).
- Ranks projects by predicted funding probability and surfaces the bottom 10% as the recommended intervention list.

Why a bottom-10% list rather than the raw threshold: at the 0.68 threshold the model flags 78,523 of the 185,103 test projects (42%) as at-risk — with precision 0.49 and recall 0.68 for the unfunded class at the model's 0.757 test ROC AUC — far more than a small review team can act on. The recommendation therefore ranks projects by predicted funding probability and hands reviewers only the lowest 10% (about 18,500 projects on a test-cohort-sized batch); the notebook omits separate precision and recall at that decile cut.

Grid search trailed the default XGBoost configuration on cross-validation score (best tuned ROC AUC 0.756 vs 0.766 for the default). The final model is reported on the held-out test set either way, and the gap is documented in the notebook.

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

## Reproducing the analysis

1. Install dependencies: `pip install -r requirements.txt`
2. Download the Kaggle data into `Datasets/` as described above.
3. Open `ML_Final.ipynb` and run top to bottom. Cross-validation and grid search are the slowest stages on the full dataset.
4. Reproduce: `jupyter notebook ML_Final.ipynb` (Run All). Estimated runtime: unmeasured end to end. The only timings the notebook logs are the grid search's 192 fits at 1.4–3.8 s each (about 7.6 minutes of summed fit time, parallelized with `n_jobs=-1`); no other stage is timed.

## Limitations and future work

- The model uses only static, at-posting features. It identifies the intervention target rather than the intervention method because causal intervention effects remain outside the model.
- A production version would pair the model with a database-backed decision support system and a reviewer-facing interface for ad hoc analysis.
- Intervention analysis, including measuring uplift from reviewer contact, is the natural next step.

## Team

Built as a team project for the Machine Learning Foundations course at Carnegie Mellon University (Spring 2025) by Iqbal, Savellano, and Zavala Reina.

## License

Released under the MIT License. See [LICENSE](LICENSE).
