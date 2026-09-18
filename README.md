# Residential Usage Prediction System

An end-to-end, production-grade machine learning prediction system developed for the **Anacity (Part of Anarock Group) AI Engineer Assignment**.

The system analyzes residential amenity booking histories to accurately anticipate resident behaviors and trigger personalized, proactive booking nudges.

---

## 1. Project Overview & Problem Statement

Modern residential gated communities offer premium shared amenities—such as gymnasiums, swimming pools, tennis courts, badminton halls, clubhouses, and multipurpose halls. However, managing these shared resources typically results in two primary friction points:
1. **Peak-Hour Crowding & Slot Contention**: Popular amenities become fully booked within minutes of opening, frustrating residents who miss out.
2. **Notification Fatigue & Spam**: Generic broadcast alerts fail to engage residents because they are neither personalized to resident habits nor timed when the resident is actually planning their week.

### The Objective
This system predicts resident usage patterns across four coupled dimensions:
1. **Which Facility** the resident is likely to use next (`facility`).
2. **Which Day of the Week** they will use it (`usage_day`).
3. **What Time / Hour** of the day they will use it (`usage_hour`).
4. **When to Send the Proactive Notification** (`notification_time` / `nudge_time`) to deliver an alert shortly before the resident's habitual booking window.

---

## 2. Approach & Architecture

Rather than treating the problem as four independent models, the system implements a **4-Stage Cascaded Gradient Boosted Decision Tree (GBDT)** architecture powered by LightGBM. Amenity booking decisions are physically and behaviorally coupled: a resident's chosen hour strongly depends on the specific facility and whether it is a weekday or weekend.

```
+-------------------------------------------------------------------------+
|                  Resident Booking History & Causal State                |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|  Stage 1: Facility Classifier (Multi-Class GBDT)                        |
|  Predicts amenity preference based on personal frequency & community    |
|  popularity priors.                                                     |
+-------------------------------------------------------------------------+
                                     |
                         Facility Probabilities
                                     v
+-------------------------------------------------------------------------+
|  Stage 2: Usage Day Classifier (7-Class GBDT)                           |
|  Predicts Day-of-Week conditioned on predicted facility & resident      |
|  weekday/weekend tendencies.                                            |
+-------------------------------------------------------------------------+
                                     |
                       Facility + Day Probabilities
                                     v
+-------------------------------------------------------------------------+
|  Stage 3: Usage Hour Classifier (24-Class GBDT)                          |
|  Predicts operating hour conditioned on facility operating bounds &     |
|  predicted day.                                                         |
+-------------------------------------------------------------------------+
                                     |
                    Facility + Day + Hour Probabilities
                                     v
+-------------------------------------------------------------------------+
|  Stage 4: Lead-Time Regressor (Quantile GBDT, alpha=0.30)               |
|  Predicts advance booking interval; conservative quantile guarantees   |
|  notification is delivered prior to habitual reservation action.        |
+-------------------------------------------------------------------------+
```

### Key Engineering Pillars

- **Strict Causal Integrity (Zero Leakage)**: Historical feature extraction operates strictly on completed past events ($t_{\text{usage}} \le t_{\text{booking}}$). Future reservations are tracked as active commitments rather than completed history, preventing negative recency anomalies.
- **Chronological Holdout Validation**: The dataset spans 6 months (January 1 – June 30, 2026). Months 1–4 are used for training, Month 5 for validation/early stopping, and Month 6 (2,473 bookings) is preserved as a strictly untouched, forward-looking test set.
- **Realistic Synthetic Dynamics**: The synthetic generator (`src/data/generator.py`) incorporates 8 physical community dynamics: resident amenity preferences, Pareto popularity distributions, facility operating hours, lead-time variations, activity sparsity, day/time clustering, operational noise, and behavioral habit drift over time.
- **Microsecond Latency**: Built on compact, optimized GBDT trees that perform full 4-stage inference in $<1.0\text{ ms}$ per resident on standard CPU hardware.

---

## 3. Technology Stack

| Layer | Technologies Used | Rationale |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.10+ | Production standard for ML pipelines |
| **Modeling & Inference** | LightGBM | Ultra-fast tabular GBDT with native categorical handling and quantile regression |
| **Preprocessing & Metrics** | scikit-learn, NumPy, pandas | Robust data transformations, circular metrics, and categorical encoding |
| **Export & Deliverables** | openpyxl, Jinja2 / HTML5 / CSS3 | Standalone review tables (.csv, .xlsx, zero-dependency .html) and executive slide deck |
| **Interactive Dashboard** | Streamlit | Responsive web UI for filtering predictions, resident inspection, and metrics review |
| **Testing & Verification** | pytest | 14 automated unit/integration tests verifying schema, causality, and metrics |

---

## 4. Deliverables Checklist (Per Assignment §4)

| Deliverable | Description | File Location |
| :--- | :--- | :--- |
| **01 Dataset** | 6-month synthetic dataset (12,909 records across 350 residents with 8 realism factors; strict 4-column schema) | [`data/facility_bookings.csv`](data/facility_bookings.csv) |
| **02 Technical Documentation** | Full architectural explanation, feature design, leakage controls, model trade-offs, holdout benchmarking, error analysis & production roadmap | [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) |
| **03 Prediction Review Output** | Prediction review table matching PDF §3.5 format (`PAST BOOKINGS` \| `PREDICTION` \| `ACTUAL` \| `MATCH`) across CSV, Excel, and interactive HTML | [`output/prediction_review_table.csv`](output/prediction_review_table.csv)<br>[`output/prediction_review_table.xlsx`](output/prediction_review_table.xlsx)<br>[`output/prediction_review.html`](output/prediction_review.html) |
| **Executive Presentation** | High-impact, standalone HTML executive slide deck highlighting architecture, metrics, and interview defenses | [`output/presentation.html`](output/presentation.html) |
| **Interview Defense & Prep Guide** | Comprehensive HTML technical brief covering decision trade-offs, class imbalance, model comparisons & top 10 interview Q&As | [`output/interview_defense_guide.html`](output/interview_defense_guide.html) |
| **Interactive Dashboard** | Real-time Streamlit dashboard for filtering predictions, inspecting resident histories, and reviewing confusion metrics | [`src/ui/app.py`](src/ui/app.py) |

---

## 5. Quickstart & Execution Guide

### Step 1: Environment Setup
Ensure Python 3.10+ is available, then install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline (One Command)
Generates the dataset, extracts leak-free causal features, trains the cascaded models and baselines, evaluates on the unseen Month 6 holdout, and exports all review deliverables:
```bash
python -m src.cli --mode all
```
*Total execution time: ~17 seconds on standard CPU.*

CLI options:
- `--mode all`: Run end-to-end pipeline (generate, train, evaluate, export).
- `--mode generate`: Regenerate `data/facility_bookings.csv`.
- `--mode train`: Retrain cascaded models and serialize to `models/artifacts/`.
- `--mode evaluate`: Evaluate trained models against unseen holdout data.
- `--mode export`: Generate `output/prediction_review_table.{csv,xlsx,html}`.

### Step 3: Launch Interactive Streamlit Dashboard
```bash
streamlit run src/ui/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to inspect resident histories, filter predictions by match criteria (All, Exact 4/4, Partial 3/4, Misses), and explore metric breakdowns.

### Step 4: View Standalone Presentation, Review Table & Interview Defense Guide
All deliverables are self-contained HTML files requiring zero server infrastructure:
- **Executive Presentation Deck**: Open `output/presentation.html` in any modern browser.
- **Interview Defense & Technical Guide**: Open `output/interview_defense_guide.html` for deep decision rationale, model comparisons, and interview Q&As.
- **Interactive Review Table**: Open `output/prediction_review.html` for instant client-side search and filtering.

### Step 5: Run Automated Test Suite
```bash
pytest tests/ -v
```
All 14 unit and integration tests run in ~2.4 seconds, validating:
- Schema constraints (4 columns matching PDF §3.1)
- Physical operating bounds and facility capacities
- Temporal causality and zero future lookahead leakage
- Model serialization and multi-stage output coupling
- Circular MAE and nudge window match logic

---

## 6. Performance Benchmarks (Unseen Month 6 Holdout)

Evaluated on **2,473 strictly chronologically unseen holdout bookings**:

| Metric / Evaluation Dimension | Cascaded GBDT (Ours) | Habitual Mode Baseline | Global Community Baseline | Impact / Lift |
| :--- | :---: | :---: | :---: | :--- |
| **Facility Top-1 Accuracy** | **46.99%** | 35.87% | 34.37% | +12.6% over Global prior |
| **Usage Day Accuracy** | **66.36%** | 21.07% | 15.65% | **+45.3% (3.1x lift)** over Habitual |
| **Usage Hour Accuracy ($\pm 1\text{h}$)** | **52.04%** | 31.42% | 23.25% | +28.8% over Global prior |
| **Nudge Window Match** | **35.22%** | 20.30% | 19.81% | +15.4% actionable lead-time alignment |
| **Exact 4 of 4 Matches** | **5.46%** (135) | **0.00%** (0) | **0.00%** (0) | Joint multi-output alignment |
| **Partial 3+ of 4 Matches** | **26.20%** (648) | 7.93% (196) | 3.44% (85) | **3.3x higher** partial accuracy |
| **Mean Outputs Matched** | **1.72 / 4.0** | 0.97 / 4.0 | 0.81 / 4.0 | +77% improvement in recommendation relevance |

---

## 7. Repository Structure

```
.
├── README.md                          # Project overview, architecture, and execution guide
├── requirements.txt                   # Pinned dependencies
├── data/
│   └── facility_bookings.csv          # Generated 6-month synthetic dataset (12,909 records)
├── docs/
│   └── TECHNICAL_DOCUMENTATION.md     # In-depth technical report and production roadmap
├── models/
│   └── artifacts/                     # Serialized LightGBM models and categorical encoders
├── output/
│   ├── metrics_summary.json           # Evaluation metrics on unseen holdout set
│   ├── presentation.html              # Standalone executive slide deck
│   ├── interview_defense_guide.html   # Comprehensive interview defense & technical guide
│   ├── prediction_review.html         # Standalone interactive HTML comparison table
│   ├── prediction_review_table.csv    # Review table matching PDF §3.5 format
│   └── prediction_review_table.xlsx   # Excel workbook with prediction and metrics sheets
├── src/
│   ├── cli.py                         # Unified CLI runner (--mode all|generate|train|evaluate|export)
│   ├── data/
│   │   ├── community_config.py        # Facility parameters, capacities, and resident archetypes
│   │   └── generator.py               # 8-factor synthetic simulation generator
│   ├── evaluation/
│   │   ├── error_analysis.py          # Cohort error analyzer (Power, Regular, Casual, Cold-Start)
│   │   └── metrics.py                 # Multi-output and composite matching metrics
│   ├── features/
│   │   ├── pipeline.py                # Causal, stateful leak-free feature extractor
│   │   └── temporal_splitter.py       # Strict chronological train/val/test splitter
│   ├── models/
│   │   ├── baselines.py               # Habitual Mode and Global Community baselines
│   │   └── cascaded_pipeline.py       # 4-stage Cascaded GBDT model pipeline
│   ├── presentation/
│   │   └── formatter.py               # PDF §3.5 table formatter and HTML exporter
│   └── ui/
│       └── app.py                     # Streamlit interactive dashboard
└── tests/
    ├── test_data_generator.py         # Tests for physical constraints & operating hours
    ├── test_evaluation.py             # Tests for circular MAE & 4-of-4 match logic
    ├── test_leakage_safety.py         # Automated assertions for zero future lookahead
    └── test_models.py                 # Model shape, fitting, and serialization tests
```

---

## Author
**Prajwal Rao**  
Personal workspace repository.
