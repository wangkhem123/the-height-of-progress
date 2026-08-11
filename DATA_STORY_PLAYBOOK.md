# Data Story Playbook
### A reusable methodology for building narrative data visualizations with ML

*Developed during VizCon 2026. Use this for any dataset where you want to go from raw numbers to an interactive story that explains WHY, not just WHAT.*

---

## The Process (7 Steps)

```
1. STATE THE FACT        → What's the shocking gap/number?
2. RESEARCH THE WHY      → What does science say drives this?
3. GET THE DATA          → Download, document sources
4. CLEAN HONESTLY        → Handle nulls transparently, drop bad features
5. MODEL & VALIDATE      → Train, test on unseen entities, flag proxies
6. BUILD THE NARRATIVE   → 6-page arc: fact → trend → experiment → model → close
7. JUDGE YOURSELF        → Score before submitting
```

---

## Step 1: State the Fact

Before any analysis, find your headline:
- What's the biggest gap in your outcome variable? (e.g., "20 cm between tallest and shortest countries")
- Who's at the extremes?
- Has it changed over time?

**Your title should BE the message.** Not "Height Data Analysis" but "The Tape Measure Never Lies." The tagline is the takeaway. The viz below proves it.

---

## Step 2: Research the Why

Adopt a research analyst mindset. Search for:
- Systematic reviews on determinants of your outcome
- RCTs and natural experiments in the domain
- Known policy levers with published effect sizes

Output: a table of factors, their evidence strength, and whether data exists.

**Rule**: Research informs which features to model — it does NOT add data to your dataset without explicit decision.

---

## Step 3: Get the Data

Create a download script that:
- Lists every source URL
- Saves to `data/raw/` with descriptive names
- Prints what was downloaded
- Is run manually (never auto-executed)

**Rule**: Only use data you explicitly downloaded. No silent web scraping mid-analysis.

---

## Step 4: Clean Honestly

Before training, produce a **Data Sanitation Report**:

```
Feature                     | Non-null % | Coverage years | Action
----------------------------|-----------|----------------|--------
protein_g_per_day           | 91%       | 1961-2021      | Keep
physician_density           | 78%       | 1960-2022      | Keep
breastfeeding_exclusive_pct | 28%       | 2000-2020      | DROP (>70% missing)
```

### Null Handling Rules
1. Let XGBoost handle NaN natively (don't median-impute)
2. Drop features with >70% missing
3. Never assign future data to past exposure windows
4. Tell the user exactly what happened

### What NOT to do
- Don't interpolate missing years silently
- Don't impute with the global median (makes poor countries look average)
- Don't use a "closest year" fallback without flagging it

---

## Step 5: Model & Validate

### Train
- XGBoost ensemble (group-separate if applicable)
- Features = only environment/policy variables (no entity labels, no outcome proxies)

### Validate with THREE strategies

| Strategy | What it tests | This is your... |
|----------|--------------|-----------------|
| Random CV | Can it fit the data? | Flattering number (context only) |
| Leave-entity-out | Can it predict unseen entities? | **Headline number** |
| Temporal split | Can it predict the future? | Forward-looking test |

### Flag proxies
If a feature's learned direction contradicts domain knowledge:
- Label it as a proxy in the model report
- Label it in the visualization
- Exclude it from "biggest lever" recommendations
- Do NOT silently flip its sign

### Natural experiment
Find matched entities that share one trait but differ in outcome:
- Same genetics, different policy (Korea split)
- Same country, different era (Netherlands 1900 vs today)

This PROVES environment matters. The model then QUANTIFIES which features matter most.

---

## Step 6: Build the Narrative

### The 6-Page Arc

| Page | Purpose | Key element |
|------|---------|-------------|
| 1. THE FACT | Shock with the gap | One big number + comparison visual |
| 2. THE TREND | Show evolution | Animated timeline with annotations |
| 3. THE DEEP DIVE | Let them explore | All entities, searchable, clickable |
| 4. THE EXPERIMENT | Prove causation | Matched comparison — the centrepiece |
| 5. THE MACHINE | Quantify drivers | Interactive model with sliders + waterfall |
| 6. THE CLOSE | Land the message | One sentence + limitations + credits |

### Design Rules
- Single self-contained HTML file (works when double-clicked)
- One accent colour only
- Three fonts maximum
- One dominant number per page
- Annotations directly on charts
- Cue lines between pages (questions pulling forward)
- Error bands visible on every prediction
- Collapsible methodology section

### Honesty in the Visual
- State: "Trained on N entities, N rows, years X-Y"
- State: "Features dropped: [list] for >70% missing"
- Show: predicted vs actual for every preset
- Show: ±RMSE error band from holdout
- Flag: proxy features with explanation
- Disclaim: "exploration tool, not a prediction engine"

---

## Step 7: Judge Yourself

Score 1–10 on each criterion before submitting:

### 1. Data Storytelling & Impact (25%)
- Is the title the message (not just a topic)?
- Does the first page hook in 10 seconds?
- Does the natural experiment provide emotional peak?
- Does the close restate the takeaway?

### 2. Discovery & Innovation (20%)
- Is there at least one counter-intuitive finding?
- Does the methodology go beyond "I plotted a chart"?
- Would an expert learn something?

### 3. Visual Design & Aesthetics (20%)
- Consistent design system throughout?
- Every animation carries meaning (not decoration)?
- Scannable in 3 seconds per page?

### 4. Data Quality & Inclusivity (20%)
- Every number traceable to a source?
- Null handling documented?
- Accessible (ARIA, focus states, reduced-motion)?
- No populations framed as inferior?

### 5. Technical Execution & Engagement (15%)
- Opens in browser, no errors?
- Interactions respond in <300ms?
- Every interaction teaches something?

### Scoring template

```
Storytelling:  _/10  Strengths:           Gaps:
Discovery:     _/10  Strengths:           Gaps:
Design:        _/10  Strengths:           Gaps:
Data Quality:  _/10  Strengths:           Gaps:
Technical:     _/10  Strengths:           Gaps:

Weighted Total: _/10

Top 3 quick wins (< 30 min each):
1.
2.
3.
```

---

## The README Template

Every project gets a README with:
1. Title + theme connection
2. The story (2-3 sentences)
3. Data sources table
4. Methodology (target, features, alignment, model, hyperparams)
5. Null handling (what was dropped, why, how)
6. Validation table (3 strategies)
7. Key findings
8. Limitations (be honest)
9. Project structure
10. How to run
11. Tools + versions
12. Future work
13. Credits + license

---

## How to Share This

- **With your team**: Drop this file in a shared repo or wiki
- **At VizCon**: Include it as a supplementary methodology document
- **As a blog post**: This is a publishable "how we built it" guide
- **As a Kiro steering file**: Copy to `.kiro/steering/data-story-playbook.md` for AI-assisted reuse

---

*Built from the VizCon 2026 Height Story project. Fork it, adapt it, make it yours.*
