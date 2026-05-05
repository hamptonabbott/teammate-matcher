# Teammate Matcher

> ML-powered student team formation using clustering and assignment optimization.

A data science final project for **DTSC 2302** at UNC Charlotte. The system takes anonymous student survey responses and produces balanced team configurations — optimizing for either schedule/work-style compatibility or complementary skill coverage, depending on what the instructor needs.

📄 **[Read the technical report →](https://hamptonabbott.com/projects/teammate-matcher-report)**
📝 **[Read the portfolio post →](https://hamptonabbott.com/projects/birds-of-a-feather)**

---

## Motivation

Random group assignment produces uneven workloads, schedule conflicts, and mismatched work styles. Existing tools rely on indirect behavioral proxies (LMS click counts, course-forum activity). This project uses **primary survey data** — actual self-reported availability, skills, and work style — to form teams more thoughtfully.

The system outputs a **set of candidate team configurations with interpretable per-metric scores**, not a final assignment. The instructor stays in the loop and makes the final call.

---

## How It Works

Two distinct matching objectives are treated as separate problems with separate feature sets:

- **Similarity** — group students with compatible schedules, work styles, and communication preferences to minimize friction
- **Complementarity** — distribute technical skills across teams so no group is uniformly weak in any single area

Four models are implemented and compared. Two of them are beyond standard introductory data-science content:

| Model | Objective | Key feature | Beyond class |
|---|---|---|---|
| K-Means | Similarity | Baseline clustering; fixed `k = 8` to match deployment scale | – |
| Agglomerative (Ward) | Similarity | Dendrogram for visual interpretation | – |
| Hungarian Assignment | Similarity + size constraint | Guarantees balanced team sizes via `scipy.optimize.linear_sum_assignment` (3–6 students per team) | ★ |
| Gaussian Mixture Model | Complementarity | Soft probabilistic assignments via EM, BIC-selected `k` | ★ |

Hungarian Algorithm is the recommended deployment model — it's the only one that guarantees equal team sizes, a practical requirement no standard clustering algorithm enforces.

---

## Dataset

Anonymous student survey deployed via Google Forms within the course. **Not** sourced from Kaggle, UCI, or any public repository — primary data collection.

- **N = 31 responses** from a single section of DTSC 2302
- **35 survey items** across 5 categories → **50 features** after preprocessing
- Survey instrument: [`survey/survey_questions.md`](survey/survey_questions.md)

| Category | Features |
|---|---|
| Schedule & availability | `avail_mon` – `avail_sun`, `avail_morning/afternoon/evening/latenight` |
| Time commitment & meeting mode | `weekly_hours`, `meeting_mode` |
| Technical skills (self-rated 1–5) | `skill_python`, `skill_data_analysis`, `skill_statistics`, `skill_visualization`, `skill_ml`, `skill_writing`, `skill_research`, `skill_presenting` |
| Work style | `role_pref`, `deadline_style`, `comm_pref`, `checkin_freq`, `collab_style`, `detail_orientation`, `conflict_style` |
| Self-assessment | `gpa_band` (optional), `contrib_*`, `pain_point` |

### Privacy

The survey was anonymous — no names, emails, or student IDs collected. Two protective measures address timestamp-based quasi-identification:

1. The raw CSV (`data/raw_survey_responses.csv`) is `.gitignore`d and stays local-only — never published.
2. The processed CSV is row-shuffled (Step 9 of the pipeline, `random_state=42`) before saving, breaking the correlation between row position and submission order.

The published `data/processed_survey_data.csv` contains no names, emails, IDs, timestamps, or ordering artifact — only encoded normalized features.

---

## Project Structure

```
teammate-matcher/
├── data/
│   ├── raw_survey_responses.csv      # ⛔ gitignored — local only
│   └── processed_survey_data.csv     # row-shuffled, anonymized
├── survey/
│   └── survey_questions.md           # full survey instrument with feature map
├── src/
│   ├── preprocess.py                 # 9-step preprocessing pipeline
│   ├── models.py                     # K-Means, Agglomerative, Hungarian, GMM wrappers
│   └── evaluate.py                   # six-metric evaluation
├── notebooks/
│   ├── 01_eda.ipynb                  # exploratory data analysis
│   ├── 02_preprocessing.ipynb        # pipeline walkthrough
│   ├── 03_models_1_2.ipynb           # K-Means + Agglomerative
│   ├── 04_models_3_4.ipynb           # Hungarian + GMM
│   └── 05_evaluation.ipynb           # comparison table + PCA biplot
├── outputs/
│   ├── pipeline_diagram.png          # methodology overview
│   ├── schedule_heatmap.png          # cohort availability matrix
│   ├── pca_biplot.png                # 8-loading PCA biplot
│   ├── comparison_metrics.png        # six-metric bar chart
│   ├── poster_comparison_table.png   # poster-ready table
│   ├── gmm_ambiguity.png             # GMM soft-assignment heatmap
│   ├── evaluation_metrics.csv        # tabular metrics
│   ├── technical_report_thumbnail.png
│   ├── portfolio_post_thumbnail.png
│   └── team_assignments/             # ⛔ gitignored
├── proposal.md                       # original early proposal (kept as history)
├── portfolio_post.md                 # public-facing write-up
├── technical_report.md               # full technical documentation
└── README.md
```

---

## Setup

```bash
git clone https://github.com/HPAuncc/teammate-matcher.git
cd teammate-matcher
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter adjustText
```

Place a Google Forms CSV export at `data/raw_survey_responses.csv` (the file is `.gitignore`d), then run notebooks in order (`01` → `05`), or run the pipeline directly:

```bash
python src/preprocess.py        # produces data/processed_survey_data.csv
```

```python
from src.preprocess import run_pipeline
from src.models     import run_all_models
from src.evaluate   import evaluate_all, print_comparison_table

df, fsets = run_pipeline()
results   = run_all_models(fsets['compatibility'], fsets['complementarity'])
eval_df   = evaluate_all(fsets['compatibility'].values,
                         fsets['complementarity'].values,
                         df, results)
print_comparison_table(eval_df)
```

All random processes are seeded with `random_state=42` for full reproducibility.

**Requirements:** Python 3.10+, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, jupyter, adjustText.

---

## Evaluation

Six metrics are reported across all four models — three algorithmic, three domain-specific — so the "best" model is explicit about what you're optimizing for:

| Metric | What it measures | Direction |
|---|---|---|
| Silhouette Score | Cluster cohesion vs. separation | ↑ higher = better |
| Davies-Bouldin Index | Avg. similarity of each cluster to its nearest neighbor | ↓ lower = better |
| Calinski-Harabasz Index | Ratio of between- to within-cluster dispersion | ↑ higher = better |
| Intra-team Skill Variance | Std dev of skill ratings within each team | ↕ context-dependent |
| Schedule Overlap | Mean Jaccard similarity of availability vectors within teams | ↑ higher = better |
| Skill Coverage | Skill dimensions where ≥1 team member scores ≥3 (normalized ≥0.5), averaged across teams (max 8) | ↑ higher = better |

Full numerical results, with interpretation, are in [`technical_report.md`](technical_report.md) §5.

---

## Selected findings

- **Hungarian Algorithm produces the most deployable configuration**: balanced 3–6-student teams, tied for highest skill coverage at 7.875 / 8.
- **Schedule, not skill, is the primary axis of student differentiation.** PCA: PC1 (15.8%) is dominated by day-of-week availability; PC2 (13.5%) by conflict-handling and meeting-mode preference. Self-rated skills don't appear in the top loadings until PC3.
- **GPA is not a passive feature.** Adjusted Rand Index between K-Means clusterings with vs. without `gpa_band` is 0.34 — moderate, not negligible. The recommended deployment protocol presents both configurations to the instructor rather than treating GPA inclusion as a default.
- **GMM converged with zero ambiguous students on this cohort** (max posterior ≥ 0.999 for all 31). The ambiguity-flag mechanism is intact for future deployments with larger or more heterogeneous cohorts.

---

## Ethics & Limitations

- **Anonymous by design.** No names, emails, or student IDs collected; raw CSV gitignored; processed CSV row-shuffled before save.
- **Self-report bias is a fairness concern.** Students from cultures where confidence-expression is discouraged may systematically under-rate themselves. Survey items are framed as *comfort* rather than *ability*; the Hungarian Algorithm's size constraint forces mixed teams; the Skill Coverage metric specifically asks whether *anyone* on the team meets each threshold.
- **Anchoring pressure.** An instructor presented with a single confident-looking algorithmic output is more likely to accept it than to scrutinize it. The system therefore surfaces *multiple* configurations and per-metric trade-offs rather than a single composite ranking.
- **Equity constraint** — a minimum-skill-diversity rule (no team composed entirely of students self-rating below threshold on any one skill dimension) is **documented as a deployment requirement but not currently enforced in code**. Implementing this is named as future work in the technical report.
- **Static snapshot.** The model captures preferences at one moment; team dynamics evolve. Validation requires a post-project satisfaction survey — also future work.
- **Sample size.** N = 31 responses from a single section limits statistical generalization, though it is sufficient for team formation within this cohort.

---

## Course Context

**DTSC 2302** — Data Science II, UNC Charlotte
Final group project · 100 points · Spring 2026

---

## AI Tool Transparency

Claude (Anthropic) was used as a coding assistant throughout development of the preprocessing pipeline, model wrappers, evaluation code, and editorial drafts of the report and portfolio post. All design decisions, research questions, feature-set construction, and ethical framing originate with the authors. Full source code was reviewed and tested against the actual dataset before inclusion.

---

## License

MIT — see [`LICENSE`](LICENSE).
