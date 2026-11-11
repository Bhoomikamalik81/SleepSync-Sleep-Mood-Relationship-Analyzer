# SleepSync: Sleep and Mood Relationship Analyzer

A machine learning feature for **Mindora**, a mental health app for students. The user answers a short questionnaire about their sleep and stress, and the model estimates how similar their answers are to students who report low mood. The result is shown as a simple wellness message with a coping tips link and a helpline.

> This is a wellness screening-style tool. It is **not** a medical diagnosis.

## Table of Contents

- [Why this feature](#why-this-feature)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [How the model is used in the app](#how-the-model-is-used-in-the-app)
- [Flask API](#flask-api)
- [Run it locally](#run-it-locally)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Ethics and safety](#ethics-and-safety)
- [Credits](#credits)

## Why this feature

Mindora already has an **Emotion Analyzer** (what the user feels, from journal text) and **Personalized Coping Suggestions**. This third feature looks at lifestyle instead of text: how a student sleeps and how stressed they feel, and how this relates to low mood. It also helps the coping suggestions, because the result tells which sleep problem is the biggest.

## Dataset

**Temporal Dynamics in Psychological Assessments**: 24,292 university students who answered standard questionnaires in 2021 (one row per student).

| File | Content |
|---|---|
| `demographic.csv` | age, gender, education, smoking, drinking |
| `phq9.csv` | PHQ-9 questionnaire (mood), 9 items |
| `gad7.csv` | GAD-7 questionnaire (anxiety), 7 items |
| `isi.csv` | Insomnia Severity Index (sleep), 7 items |
| `pss.csv` | Perceived Stress Scale, 14 items |

- Source: Zenodo, DOI [10.5281/zenodo.10423537](https://zenodo.org/records/10423537)
- Paper: Su et al. (2024), *Temporal dynamics in psychological assessments: a novel dataset with scales and response times*, Scientific Data, DOI 10.1038/s41597-024-03888-8
- License: CC BY 4.0 (credit is required)
- The data files are **not** included in this repository. Download them from Zenodo and place them in a folder named `dataset/`.

## Approach

All steps are in `Sleep_Mood_Analysis.ipynb`.

1. **Load and merge** the five files on `export_id`.
2. **Understand and clean the data:** data types, missing values, duplicates, outliers, score checks, and removal of impossible ages (below 16 or above 40). 24,252 students remain.
3. **Create the target:** `low_mood = 1` when the mood score is 5 or more (mild or higher mood symptoms on the PHQ-9 scale). About 25.6% of students are marked as low mood.
4. **Avoid data leakage:** PHQ-9 question 3 asks about sleep, so it is removed from the mood score. Otherwise the model would partly predict sleep from sleep.
5. **EDA:** sleep problem level vs mood, anxiety and stress, and a correlation heatmap.
6. **Sleep-only models:** Baseline, Logistic Regression, Decision Tree and Random Forest.
7. **Model improvement:** 5-fold cross-validation, hyperparameter tuning and threshold analysis.
8. **Add stress questions:** four PSS questions are chosen with forward selection (see below).
9. **Final model, insights and demo:** coefficients, a Mindora demo function, saving the model.

**Why threshold 5 and not the usual 10?** The usual PHQ-9 cutoff of 10 flags only about 4% of students, which is too few for a model to learn from.

**Why only some PSS questions?** The positively worded PSS questions (4, 5, 6, 7, 9, 10 and 13) are left out. In this data a higher answer on them goes with more low mood symptoms, which is the opposite of what the wording means, so they would confuse users. The questions selected are `pss_1`, `pss_2`, `pss_3` and `pss_14`.

## Results

**Sleep-only model (test set):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 76.48% | 53.26% | 67.10% | 59.38% | 0.806 |
| Random Forest | 74.97% | 50.86% | 69.35% | 58.68% | 0.808 |
| Decision Tree | 73.63% | 49.00% | 70.80% | 57.91% | 0.797 |
| Baseline (always "not low") | 74.38% | 0.00% | 0.00% | 0.00% | 0.500 |

Tuning and threshold changes did not help. Every model stays at ROC-AUC 0.81, and accuracy only goes up when recall goes down (at threshold 0.7: accuracy 79.63% but recall only 46.50%, so more than half of the low mood students would be missed). The seven sleep questions alone hold only a limited amount of information about mood.

**Adding stress questions (5-fold cross-validation):**

| Features | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Sleep only (7 questions) | 77.08% | 68.11% | 60.35% | 0.815 |
| Sleep + 2 stress questions | 79.95% | 77.33% | 66.39% | 0.871 |
| Sleep + 4 stress questions | 80.59% | 79.07% | 67.60% | 0.878 |

**Final model: Logistic Regression** on 7 sleep questions + 4 stress questions + age, gender, smoking and drinking (test set):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 80.46% | 58.90% | 78.52% | 67.31% | 0.875 |

Recall matters most here, because in a wellness app it is worse to miss a student who is struggling than to show a false alarm. The baseline reaches 74% accuracy without finding a single low mood student, so accuracy alone is misleading.

**Note:** the four stress questions carry about 1.5 times the weight of the seven sleep questions together (sleep share 39.4%). So the final model is a sleep and stress model, and sleep is not the only driver. The sleep questions stay in the app because they show *which* sleep problem is linked with low mood.

## How the model is used in the app

The user answers **11 questions** and gives age, gender, smoking and drinking.

| Part | Questions | Answers |
|---|---|---|
| Sleep (ISI) | `isi_1` to `isi_7` | 0 to 4 |
| Stress (PSS) | `pss_1`, `pss_2`, `pss_3`, `pss_14` | 1 to 5 |
| About you | `age`, `gender`, `smoke`, `drink` | see `model_info.json` |

The model returns a **risk score from 0 to 100**. This is a risk score, not a real probability, because class weights were used in training. It is split into three levels:

| Level | Score |
|---|---|
| Steady | below 40 |
| Disturbed | 40 to 59 |
| Needs attention | 60 and above |

Checked on the students in the dataset (the same data the model was trained on): 57.8% of students are "steady" and only 6% of them have low mood symptoms. 27.5% are "needs attention" and 65.8% of them have low mood symptoms. The levels are stored in `model_info.json`.

## Flask API

`app.py` loads `best_model.pkl` and `model_info.json`. The API does not store any answers.

| Endpoint | Description |
|---|---|
| `GET /health` | Check that the API is running |
| `GET /questions` | Questions, answer ranges, allowed options, risk levels and helpline |
| `POST /predict` | Send the answers, get the risk score and message |

**Request** (`POST /predict`):

```json
{
  "isi_1": 3, "isi_2": 2, "isi_3": 1, "isi_4": 3, "isi_5": 2, "isi_6": 3, "isi_7": 2,
  "pss_1": 3, "pss_2": 3, "pss_3": 4, "pss_14": 3,
  "age": 20, "gender": "female", "smoke": "never smokes", "drink": "never drinks"
}
```

**Response:**

```json
{
  "risk_score": 100,
  "level": "needs_attention",
  "title": "Your sleep and stress need some attention",
  "message": "Your answers are similar to students who often report low mood. Try a fixed sleep time this week and look at the coping tips.",
  "biggest_sleep_concern": { "question": "isi_1", "text": "Difficulty falling asleep" },
  "support_message": "If you have felt low for a while, consider talking to someone you trust or a counsellor.",
  "helpline": { "name": "Tele-MANAS", "number": "14416", "number_alternative": "1-800-891-4416",
                "hours": "24x7, free and confidential", "emergency_number": "112" },
  "disclaimer": "This is a wellness check, not a medical diagnosis. The score shows how your answers compare with other students."
}
```

Wrong input (missing question, answer out of range, text instead of a number, unknown gender and so on) returns status `400` with a list of `details`.

## Run it locally

```bash
# 1. clone and install
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. put the five dataset CSV files in dataset/ and run the notebook
jupyter notebook Sleep_Mood_Analysis.ipynb
#    this creates cleaned_data.csv, best_model.pkl, model_info.json and model_results.csv

# 3. start the API
python app.py                      # http://127.0.0.1:5000

# 4. in another terminal, run the automatic tests
python test_api.py
```

`test_api.py` checks `/health`, `/questions`, valid and invalid `/predict` requests and CORS. It prints PASS or FAIL for every check and exits with code 1 if anything fails.

The frontend gets the questions, answer ranges and options from `GET /questions` and the result from `POST /predict`.

## Project structure

```
SleepSync/
├── dataset/                       # the five dataset files (download from Zenodo, not included here)
│   ├── demographic.csv
│   ├── gad7.csv
│   ├── isi.csv
│   ├── phq9.csv
│   └── pss.csv
├── Sleep_Mood_Analysis.ipynb      # full workflow: data, models, final model, demo
├── app.py                         # Flask API
├── test_api.py                    # automatic API tests (PASS / FAIL)
├── best_model.pkl                 # trained model (created by the notebook)
├── model_info.json                # features, answer ranges, options, risk levels (created by the notebook)
├── cleaned_data.csv               # cleaned data (created by the notebook)
├── model_results.csv              # final model comparison (created by the notebook)
├── requirements.txt               # libraries for the notebook and the API
├── .gitignore
└── README.md
```

## Limitations

- Each student has only **one row**, so the model shows a relationship across students and not day by day changes.
- Most students are around 18 to 22 years old with a bachelor's degree, so results may not apply to other groups.
- The four stress questions were selected on the same data that was used for scoring, so the real accuracy may be slightly lower.
- A link between sleep, stress and mood does **not** prove that one causes the other.
- The `low_mood` flag means mild or higher symptoms on a questionnaire. It is not a clinical label.
- The answer labels of the PSS questions (1 to 5, "Never" to "Very often") should be confirmed against the dataset paper before the final app text is fixed.

## Ethics and safety

- Mindora must never say "you have depression". Use calm wording such as "similar to students who report low mood".
- Always show the disclaimer: *wellness check, not a medical diagnosis*.
- Show the helpline on every result, not only on "needs attention". The API sends it in every response: **Tele-MANAS 14416** (free, 24x7) and **112** for immediate danger. Please confirm the number on the official Tele-MANAS page before launch. If the app is used outside India, show a helpline for that country.
- Sleep, stress and mood answers are sensitive. Do not store them without clear consent, and do not log them.
- Restrict `ALLOWED_ORIGIN` to your own frontend address when the API is deployed.

## Credits

- Dataset: Su et al. (2024), Scientific Data, DOI 10.1038/s41597-024-03888-8. Data on Zenodo, DOI 10.5281/zenodo.10423537, CC BY 4.0.
- Questionnaires: PHQ-9, GAD-7, Insomnia Severity Index (ISI), Perceived Stress Scale (PSS-14).
- Built with Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Flask and Jupyter Notebook.
