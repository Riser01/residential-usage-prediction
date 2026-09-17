# Facility Usage Prediction System — Architectural & Technical Design Document

**Author**: Prajwal Rao  
**Target Organization**: Anacity (Part of Anarock Group)  
**Role**: Senior AI Engineer Assignment  
**Status**: DESIGN & PLANNING PHASE  
**Version**: 1.0.0  

---

## 1. Executive Summary & Business Context

Anacity operates an enterprise SaaS platform across 125+ cities, 6,800+ communities, and 700,000+ households. In residential and commercial gated communities, managing shared amenities—such as fitness centers, swimming pools, tennis and badminton courts, clubhouses, and multipurpose event halls—presents operational and resident satisfaction challenges:
- Peak-hour congestion vs off-peak under-utilization
- Facility booking friction and forgotten reservation deadlines
- Static or spammy broadcast notifications that residents ignore

### Business Objective
Build an intelligent, personalized **Facility Usage Prediction System** that analyzes historical booking behaviors to anticipate:
1. **Which facility** a resident will book next (`facility`)
2. **Which day** they will use it (`usage_day`)
3. **What time** of day they will use it (`usage_hour`)
4. **When to send a proactive notification/nudge** (`notification_time`) to maximize convenience, prevent missed bookings, and balance amenity load.

---

## 2. Core Technical Requirements & Scope

According to the Anacity prediction system specification, the core deliverables are:
1. **Working Prototype & Synthetic Dataset**: A realistic, reproducible generator simulating community dynamics (preferences, popularity, lead times, sparsity, imbalance, noise, and concept drift) and an end-to-end ML training/evaluation pipeline.
2. **Technical Documentation**: Detailed documentation explaining feature engineering, leakage prevention, modeling choices, alternatives considered, holdout evaluation results, error analysis, and production limitations.
3. **Prediction Review Output**: A transparent tabular comparison (spreadsheet and interactive UI) comparing predictions with chronologically holdout actual test data, highlighting exact match/error indicators across the four outputs.

---

## 3. Synthetic Dataset Generation Engine (`community_simulator`)

### 3.1 Community Profile
- **Community**: "Skyline Oasis Residences" (1 residential community with 350 active units / residents: `R-101` to `R-450`).
- **Time Horizon**: 6 months of historical activity (e.g., Jan 1, 2026 to Jun 30, 2026), yielding ~12,000 booking transactions.
- **Facilities & Capacity Parameters**:
  - `Gym`: Capacity 40, open 06:00-22:00. High volume, habitual routine. Typical lead time: 2–12 hours.
  - `Swimming Pool`: Capacity 25, open 06:00-21:00. Weather/seasonal sensitivity, weekend afternoon bias. Typical lead time: 4–18 hours.
  - `Badminton Court`: Capacity 4 (2 indoor courts), open 06:00-22:00. Highly competitive slots, evening/weekend focus. Typical lead time: 18–28 hours.
  - `Tennis Court`: Capacity 4 (1 outdoor court), open 06:00-11:00, 16:00-21:00. Weather-dependent, morning/evening slots. Typical lead time: 24–48 hours.
  - `Clubhouse Lounge`: Capacity 30, open 10:00-23:00. Social gatherings, weekend afternoon/evening. Typical lead time: 1–3 days (24–72 hours).
  - `Multipurpose Hall`: Capacity 150 (single event venue), open 09:00-23:00. Event-based, rare usage. Typical lead time: 3–14 days (72–336 hours).

### 3.2 Required Realism Dimensions (Per Assignment §3.1)
1. **Resident Preferences & Personas**:
   - *Persona 1: Dawn Athletes (25%)* — Gym or Tennis, 06:00–08:00 on weekdays, high regularity.
   - *Persona 2: Evening Sports Enthusiasts (25%)* — Badminton or Gym, 18:00–21:00 weekdays, weekend afternoons.
   - *Persona 3: Weekend Leisure Families (20%)* — Pool, Clubhouse, 10:00–18:00 Saturday/Sunday.
   - *Persona 4: Sporadic / Occasional Users (20%)* — Low frequency, exploratory facility choices.
   - *Persona 5: Event Organizers (10%)* — Multipurpose Hall, Clubhouse, booked far in advance.
2. **Facility Popularity (Pareto Imbalance)**:
   - Gym: ~38%, Swimming Pool: ~24%, Badminton: ~18%, Tennis: ~11%, Clubhouse: ~6%, Multipurpose Hall: ~3%.
3. **Time Patterns (Diurnal & Weekly Cycles)**:
   - Weekday peaks at 07:00 and 19:00; weekend peaks shift to 10:00–12:00 and 16:00–19:00.
4. **Booking Lead Times**:
   - Modeled via Log-Normal distributions per facility and resident archetype ($L \sim \text{Lognormal}(\mu_{f}, \sigma_{f}^2)$).
5. **Sparsity & Imbalance**:
   - Power-law resident engagement (top 20% users generate 65% of bookings; bottom 40% generate <10%).
6. **Noise & Stochasticity**:
   - 6% random exploratory bookings where residents book facilities or times outside their standard preference profile.
7. **Changing Behaviour (Concept Drift)**:
   - At month 3.5 (~April 15), 15% of residents undergo drift:
     - Shift from Gym to Swimming Pool (seasonal summer drift).
     - Shift workout times due to office/commute changes (e.g., morning 07:00 to evening 19:00).
     - New resident move-ins (cold-start residents appearing after Month 3).

---

## 4. Feature Engineering & Strict Leakage Controls

### 4.1 Temporal Integrity & Formulation
For any resident $r$ and their target $(k+1)$-th booking:
- The prediction cutoff timestamp $t_{\text{cutoff}}^{(k+1)}$ is set strictly at the usage/booking timestamp of booking $k$.
- **Zero Future Leakage Principle**: All aggregations, statistics, recency counts, and ratios are computed strictly on bookings where $usage\_timestamp < t_{\text{cutoff}}$ and $booking\_timestamp < t_{\text{cutoff}}$.
- **Zero Target Leakage Principle**: Under no circumstances is any metadata from the target booking (such as its actual booking timestamp, actual lead time, or actual day) fed into the feature set.

### 4.2 Feature Taxonomy
1. **User Historical Preference Profile (Expanding Temporal Window)**:
   - Historical frequency distribution over facilities: $P(F = f \mid \text{history}_r)$.
   - Historical frequency distribution over days of week: $P(D = d \mid \text{history}_r)$.
   - Historical frequency distribution over hours: $P(H = h \mid \text{history}_r)$.
   - User facility entropy (measure of habit vs exploration): $H(r) = -\sum p_i \log p_i$.
   - Total historical bookings count of resident $r$.
2. **Sequential & Markov Transition Features**:
   - Previous facility booked ($F_{k}$).
   - Previous usage day of week ($D_k$) and usage hour ($H_k$).
   - Facility transition probability: $P(F_{k+1} \mid F_k)$.
   - Inter-booking cadence: Days since last booking, days since last usage.
   - Day-of-week transition pattern (e.g. Mon $\to$ Wed $\to$ Fri cadence).
3. **Lead Time Historical Profile**:
   - Resident's empirical median lead time for facility $f$: $\text{median\_lead}_{r, f}$.
   - Resident's overall lead time standard deviation $\sigma_{\text{lead}, r}$.
   - Facility population median lead time: $\text{median\_lead}_f$.
4. **Community & Global Context Features**:
   - Facility congestion rate at proposed time slots (historical load factor).
   - Global facility popularity ranking in the last 30 days.
5. **Cyclical Temporal Encodings**:
   - Hour sine/cosine: $\sin(2\pi h / 24), \cos(2\pi h / 24)$.
   - Day-of-week sine/cosine: $\sin(2\pi d / 7), \cos(2\pi d / 7)$.
   - Month sine/cosine: $\sin(2\pi m / 12), \cos(2\pi m / 12)$.
6. **Cold-Start / Empirical Bayes Shrinkage**:
   - For users with $< 3$ bookings, individual frequencies are smoothed toward community global facility and slot priors:
     $$\hat{P}(f \mid r) = \frac{N_r \cdot P_{\text{sample}}(f \mid r) + M \cdot P_{\text{global}}(f)}{N_r + M}$$
     where $M = 5$ is the empirical shrinkage pseudo-count.

---

## 5. Modeling Architecture: Cascaded Multi-Target Pipeline

### 5.1 Formulation of the 4 Targets
1. **Target 1: `facility`**: Multi-class classification ($K = 6$ facilities).
2. **Target 2: `usage_day`**: Multi-class classification ($K = 7$ days: Mon...Sun).
3. **Target 3: `usage_hour`**: Multi-class classification ($K = 18$ discrete operating hours: 06:00 to 23:00).
4. **Target 4: `notification_time` (`nudge_time`)**: Calculated as:
   $$\text{Predicted Nudge Time} = \text{Predicted Usage Slot} - \text{Predicted Lead Time}$$
   Formatted as Day-of-Week + HH:MM (e.g., `Thu 18:15`).

### 5.2 Architectural Trade-off Analysis
- **Alternative A: 4 Independent Classifiers/Regressors**
  - *Pros*: Simple, embarrassingly parallel.
  - *Cons*: Ignores strong joint distribution coupling (e.g., Multipurpose Hall is never used at 07:00; Pool is heavily weekend midday). Leads to inconsistent joint predictions.
- **Alternative B: Monolithic Multi-Task Neural Network / TabNet**
  - *Pros*: Shared representation.
  - *Cons*: Heavy dependencies (PyTorch/CUDA), difficult to tune, prone to negative transfer, slower inference, lacks tabular tree interpretability (SHAP).
- **Alternative C: Cascaded Gradient Boosted Decision Pipeline (Selected Approach)**
  - *Step 1*: Predict `facility` via LightGBM multi-class classifier using historical user + community features.
  - *Step 2*: Predict `usage_day` via LightGBM multi-class classifier, taking predicted `facility` probabilities as conditional inputs.
  - *Step 3*: Predict `usage_hour` via LightGBM multi-class classifier, taking predicted `facility` and `usage_day` as inputs.
  - *Step 4*: Predict `lead_time_hours` via LightGBM Regressor (or quantile regression to optimize nudge timeliness), conditioned on predicted facility and usage slot.
  - *Step 5*: Compute `nudge_time` from predicted usage timestamp minus predicted lead time, snapping to habitual booking notification windows.
  - *Benefits*: SOTA tabular performance, calibrated probabilities, full interpretability, modular unit testing, fast CPU inference (<10ms per inference).

---

## 6. Evaluation Framework & Chronological Holdout Strategy

### 6.1 Split Strategy
- **Training Set**: Chronological Months 1 to 4 (e.g., Jan 1 to Apr 30)
- **Validation Set**: Chronological Month 5 (e.g., May 1 to May 31) — used for hyperparameter tuning and calibration.
- **Unseen Holdout Test Set**: Chronological Month 6 (e.g., Jun 1 to Jun 30) — strictly held out until final evaluation.

### 6.2 Per-Output Evaluation Metrics
1. **Facility Prediction**:
   - Multi-class Accuracy
   - Top-2 Accuracy
   - Macro F1 and Weighted F1 Scores
   - Full Confusion Matrix
2. **Usage Day Prediction**:
   - Day Accuracy (exact day of week match)
   - Top-2 Day Accuracy
   - Macro F1
3. **Usage Hour Prediction**:
   - Exact Hour Match Accuracy
   - Within $\pm 1$ Hour Accuracy
   - Mean Absolute Error (hours) and Circular MAE
4. **Notification / Nudge Time Prediction**:
   - Mean Absolute Lead Error (hours)
   - Nudge Timeliness Rate (% of nudges sent before actual booking timestamp, ensuring the nudge is actionable and not late)
   - Habitual Nudge Window Accuracy (within 60 minutes of actual booking decision)

### 6.3 Overall Composite Measure & Match Formulation
- **Exact Match (4 of 4)**: Exact match on `facility` AND `usage_day` AND `usage_hour` AND `nudge_time` (within tolerance window).
- **Partial Match (3 of 4, 2 of 4, 1 of 4)**: Capturing partial utility.
- **Composite Score**: Harmonic / Weighted accuracy score:
  $$\text{Score}_{\text{composite}} = 0.35 \cdot \text{Acc}_{\text{facility}} + 0.25 \cdot \text{Acc}_{\text{day}} + 0.20 \cdot \text{Acc}_{\text{hour}} + 0.20 \cdot \text{Acc}_{\text{nudge}}$$

### 6.4 Error Analysis Dimensions
- Breakdown by User Activity Tier: Heavy (>20 bookings), Medium (5–20 bookings), Sporadic/Cold-start (<5 bookings).
- Breakdown by Facility Type: High volume (Gym) vs rare event (Multipurpose Hall).
- Concept Drift Sensitivity: Pre-drift vs post-drift performance.

---

## 7. Deliverables & User Experience

1. **Synthetic Dataset**: `data/facility_bookings.csv` with core schema (`resident_id`, `facility_id`, `booking_timestamp`, `usage_timestamp`).
2. **Reproducible Pipeline**:
   - `python -m src.cli --mode generate`
   - `python -m src.cli --mode train`
   - `python -m src.cli --mode evaluate`
   - `python -m src.cli --mode all`
3. **Technical Documentation**: `docs/TECHNICAL_DOCUMENTATION.md` detailing decisions, mathematics, leakage controls, benchmarking, and limitations.
4. **Prediction Review Output**:
   - CSV: `output/prediction_review_table.csv`
   - Excel: `output/prediction_review_table.xlsx`
   - Standalone zero-dependency HTML: `output/prediction_review.html`
   - Interactive Streamlit Dashboard: `src/ui/app.py`
   Formatted exactly per PDF Section 3.5:
   `PAST BOOKINGS` | `PREDICTION` | `ACTUAL` | `MATCH` ("YES 4 of 4", "NO 2 of 4") with filtering, search, and metrics summary.
