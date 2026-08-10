# Height: The Most Honest Progress Report a Nation Can Produce

**VizCon 2026 | Height Story — All Countries**

---

## The Thesis

You don't need 50 indicators to measure a country's development. You need one: **how tall are its children growing up.**

We proved that height IS a development metric. A country's average height tells you — in one number — how well it fed, healed, and protected its children over the last generation.

## How We Proved It

We trained an ML model on **196 countries** using **11 childhood development indicators** — healthcare, nutrition, sanitation, equality, environment. No genetic data. No country labels. The model doesn't know *which* country it's looking at — only the conditions children grew up in.

**Result: R² = 0.96** — those 11 development indicators explain 96% of why some nations are taller than others.

Height doesn't just *correlate* with development. Height **IS** development, encoded in biology.

## Model Performance

| Model | N | Features | R² | RMSE | MAE |
|-------|---|----------|-----|------|-----|
| Male | 6,076 | 11 | 0.960 | ±1.0 cm | 0.68 cm |
| Female | 6,076 | 11 | 0.958 | ±0.9 cm | 0.60 cm |
| **Average** | **12,152** | **11** | **0.959** | **±0.9 cm** | **0.64 cm** |

Both models independently converge on the same drivers — confirming the signal is real.

## What Height Encodes (Top Features)

| Rank | Development indicator | Male | Female | Avg | What it tells you |
|------|----------------------|------|--------|-----|-------------------|
| 1 | Physician density | 41% | 33% | 37% | Did children have healthcare access? |
| 2 | Protein (g/day) | 12% | 13% | 13% | Were children fed enough protein? |
| 3 | Breastfeeding % | 9% | 11% | 10% | Did infants get proper early nutrition? |
| 4 | Healthcare spending | 8% | 7% | 7.5% | Did the country invest in health? |
| 5 | Inequality (Gini) | 6% | 7% | 6.5% | Did growth reach everyone? |
| 6 | Air pollution (PM2.5) | 4% | 6% | 5% | Was the environment safe to grow in? |
| 7 | GDP per capita | 4% | 5% | 4.5% | Was there wealth? (less important than how it's spent) |
| 8 | Caloric intake | 4% | 5% | 4.4% | Was there enough food? |
| 9 | Sanitation | 4% | 4% | 3.8% | Was disease prevented? |
| 10 | Urbanization | 4% | — | 3.5% | Did infrastructure reach people? |
| 11 | Animal protein % | 3% | 5% | 4% | Was protein quality high? |

**Key insight:** GDP ranks only #7. Being wealthy doesn't make a nation tall — *investing* that wealth in healthcare, food systems, and equality does.

## The Proof: Same Genes, Different Heights

If height were genetic destiny, populations with identical DNA would be the same height. They're not:

- **South Korea vs North Korea** — same people, split in 1945. By 1996: South Korean men are 3–4 cm taller. The only difference: development.
- **Netherlands vs Germany** — near-identical ancestry. Dutch invested in dairy + universal healthcare → now the world's tallest.

These natural experiments confirm what the model quantifies: **height tracks development, not genetics.**

## The Conclusion

> Height is a biological receipt for a generation of policy decisions.
>
> Every centimeter tells a story: Did the government invest in healthcare?
> Did protein reach children's plates? Did inequality lock people out?
>
> Measure the children. You'll know the answer.

---

## Data Sources

- **Height:** NCD-RisC, Lancet 2020 (196 countries, 1900–2015)
- **Development indicators:** World Bank, Our World in Data, FAO
- **Method:** XGBoost ensemble (5-fold), separate male/female models, childhood-window feature averaging (birth → birth+10 years)
