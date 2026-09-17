# Anacity Facility Usage Prediction System

An end-to-end, production-grade AI/ML prediction system developed for the **Anacity (Part of Anarock Group) AI Engineer Assignment**.

The system forecasts resident facility booking behavior:
1. **Which facility** a resident is likely to use next (`facility`)
2. **Which day** they are likely to use it (`usage_day`)
3. **What time** of day they will use it (`usage_hour`)
4. **When a proactive notification should be sent** (`notification_time` / `nudge_time`)

---

## Deliverables Checklist (Per Assignment §4)

| Item | Description | Location in Repo |
| :--- | :--- | :--- |
| **01 Dataset** | 6-month synthetic dataset (12,909 bookings across 350 residents with 8 realism factors) | [`data/facility_bookings.csv`](data/facility_bookings.csv) |
| **02 Technical Documentation** | Full architectural explanation, leakage controls, modeling trade-offs, holdout benchmarking, error analysis & limitations | [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) |
| **03 Prediction Review Output** | Spreadsheet and Interactive UI comparison table matching PDF §3.5 format | [`output/prediction_review_table.csv`](output/prediction_review_table.csv), [`output/prediction_review_table.xlsx`](output/prediction_review_table.xlsx), [`output/prediction_review.html`](output/prediction_review.html), and [`src/ui/app.py`](src/ui/app.py) |

---

## Quickstart & Single-Command Reproduction

### 1. Environment Setup
Activate your Python 3.10+ environment and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (One Command)
Generates the synthetic dataset, splits chronologically, extracts leak-free features, trains the cascaded GBDT model and baselines, evaluates on unseen holdout data, and exports all review deliverables:
```bash
python -m src.cli --mode all
```
*Total execution time: ~17 seconds on standard CPU.*

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run src/ui/app.py
```

### 4. View Standalone HTML Comparison Viewer
Open `output/prediction_review.html` in any web browser. It requires zero external servers or network connections and features live client-side search and filtering by match status.

### 5. Run Automated Test Suite
```bash
pytest tests/ -v
```
*All 14 unit and integration tests verify schema constraints, physical causality, zero temporal leakage, model shapes, and evaluation metrics in ~2.4 seconds.*

---

## Performance Summary (Unseen Month 6 Holdout)

Evaluated on **2,473 strictly chronologically unseen holdout bookings**:

| Metric / Output | Value | Benchmark Lift over Baseline |
| :--- | :---: | :---: |
| **Output 1: Facility Accuracy** | **46.99%** | +12.6% vs Global Community Prior |
| **Output 2: Usage Day Accuracy** | **66.36%** | **+45.3% (3.1x lift)** vs Habitual Baseline |
| **Output 3: Usage Hour (±1h)** | **52.04%** | +28.8% vs Global Community Prior |
| **Output 4: Nudge Window Match** | **35.22%** | High actionability window match |
| **Overall: Exact 4 of 4 Matches** | **5.46%** | 135 exact joint matches (vs 0% for baselines) |
| **Overall: Partial 3+ of 4 Matches** | **26.20%** | 648 records matching 3 or 4 outputs |
| **Overall: Average Outputs Matched**| **1.72 / 4.0** | +77% higher than Habitual Mode baseline |

---

## Repository Structure

```
.
├── DESIGN.md                          # Comprehensive architectural and system design document
├── TRACKER.md                         # Milestone implementation and verification tracker
├── ADVERSARIAL_REVIEWS.md             # 5-Reviewer Adversarial Panel review records
├── README.md                          # This setup and verification guide
├── requirements.txt                   # Pinned dependency requirements
├── data/
│   └── facility_bookings.csv          # Generated 6-month synthetic dataset (12,909 records)
├── docs/
│   └── TECHNICAL_DOCUMENTATION.md     # In-depth technical report and production roadmap
├── models/
│   └── artifacts/                     # Serialized LightGBM models and categorical encoders
├── output/
│   ├── metrics_summary.json           # Evaluation metrics on unseen holdout set
│   ├── prediction_review.html         # Standalone zero-dependency HTML comparison table
│   ├── prediction_review_table.csv    # Raw review table matching PDF §3.5 format
│   └── prediction_review_table.xlsx   # Excel workbook with prediction and metric sheets
├── src/
│   ├── cli.py                         # Unified CLI runner (--mode all|generate|train|evaluate|export)
│   ├── data/
│   │   ├── community_config.py        # Facility parameters, capacities, and resident archetypes
│   │   └── generator.py               # 8-factor synthetic simulation generator
│   ├── evaluation/
│   │   ├── error_analysis.py          # Cohort error analyzer (Power, Regular, Cold-start)
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
│       └── app.py                     # Streamlit web dashboard
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
