# Reddit Mental Health Classification — Full Evaluation Results (Version 8)

> **Experiment Configuration**: `run_version_number = 8`, `task = binary`, `subsample = 200` (400 samples total per run: 200 target condition + 200 balanced control pool), `model = 0` (SGDClassifier, L1 penalty, log_loss, alpha=0.005, random_state=42).

---

## 1. Executive Master Summary

Comparison of all 15 mental health conditions across three evaluation benchmarks:
1. **In-Domain Test Set** (80 samples held-out, 20% test split from pre-pandemic data)
2. **Mid-Pandemic Same-Condition Transfer Test** (March 11 – April 20, 2020 data for the same condition vs control)
3. **COVID-19 Cross-Domain Generalization Test** (`COVID19_support` post-pandemic data)

| Subreddit | In-Domain Acc | In-Domain W-F1 | Control F1 | Target F1 | Mid-Pandemic Acc | Mid-Pandemic W-F1 | COVID-19 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **addiction** | 0.875 | 0.875 | 0.872 | 0.878 | 0.961 | 0.960 | 0.537 |
| **EDAnonymous** | 0.825 | 0.825 | 0.821 | 0.829 | 0.850 | 0.848 | 0.514 |
| **adhd** | 0.925 | 0.925 | 0.919 | 0.930 | 0.950 | 0.950 | 0.441 |
| **autism** | 0.825 | 0.824 | 0.829 | 0.821 | 0.936 | 0.936 | 0.500 |
| **alcoholism** | 0.925 | 0.925 | 0.914 | 0.933 | 0.919 | 0.919 | 0.560 |
| **bipolarreddit** | 0.925 | 0.925 | 0.914 | 0.933 | 0.934 | 0.934 | 0.474 |
| **bpd** | 0.887 | 0.888 | 0.877 | 0.897 | 0.893 | 0.893 | 0.403 |
| **depression** | 0.838 | 0.838 | 0.854 | 0.817 | 0.904 | 0.904 | 0.425 |
| **anxiety** | 0.800 | 0.801 | 0.795 | 0.805 | 0.974 | 0.974 | 0.412 |
| **ptsd** | 0.825 | 0.826 | 0.837 | 0.811 | 0.859 | 0.859 | 0.397 |
| **healthanxiety** | 0.950 | 0.950 | 0.955 | 0.944 | 0.912 | 0.912 | 0.459 |
| **lonely** | 0.812 | 0.811 | 0.839 | 0.776 | 0.867 | 0.867 | 0.488 |
| **schizophrenia** | 0.900 | 0.900 | 0.909 | 0.889 | 0.961 | 0.960 | 0.444 |
| **socialanxiety** | 0.825 | 0.824 | 0.848 | 0.794 | 0.903 | 0.903 | 0.500 |
| **suicidewatch** | 0.912 | 0.913 | 0.921 | 0.901 | 0.938 | 0.937 | 0.552 |

---

## 2. Psychological Profiling: Cross-Domain Transfer to COVID-19

When evaluating how often pandemic support posts (`COVID19_support`) were classified as each respective mental health condition:

| Subreddit Condition | COVID-19 Support Classification Rate | Interpretation |
| :--- | :---: | :--- |
| **addiction** | 6.1% | Low/distinct symptom signature |
| **EDAnonymous** | 12.2% | Low/distinct symptom signature |
| **adhd** | 14.7% | Low/distinct symptom signature |
| **autism** | 14.7% | Low/distinct symptom signature |
| **alcoholism** | 25.0% | Elevated overlap with pandemic distress |
| **bipolarreddit** | 5.1% | Low/distinct symptom signature |
| **bpd** | 9.7% | Low/distinct symptom signature |
| **depression** | 12.5% | Low/distinct symptom signature |
| **anxiety** | 14.7% | Low/distinct symptom signature |
| **ptsd** | 19.1% | Low/distinct symptom signature |
| **healthanxiety** | 9.5% | Low/distinct symptom signature |
| **lonely** | 8.3% | Low/distinct symptom signature |
| **schizophrenia** | 11.1% | Low/distinct symptom signature |
| **socialanxiety** | 8.6% | Low/distinct symptom signature |
| **suicidewatch** | 12.1% | Low/distinct symptom signature |

---

## 3. Detailed Per-Condition Breakdown

### `addiction`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.791 | 0.971 | 0.872 | 35 |
| **addiction** | 0.973 | 0.800 | 0.878 | 45 |
| **accuracy** | 0.875 | 0.875 | 0.875 | - |
| **macro avg** | 0.882 | 0.886 | 0.875 | 80 |
| **weighted avg** | 0.893 | 0.875 | 0.875 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `goals` | +10.198 | `and` | -9.730 |
| `recovery` | +10.170 | `after` | -6.895 |
| `regularly` | +8.612 | `my` | -6.060 |
| `support meetings` | +8.612 | `to` | -5.152 |
| `staying` | +8.212 | `in` | -5.043 |

#### Mid-Pandemic Evaluation: Accuracy = **0.961**, Weighted F1 = **0.960**

---

### `EDAnonymous`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.744 | 0.914 | 0.821 | 35 |
| **EDAnonymous** | 0.919 | 0.756 | 0.829 | 45 |
| **accuracy** | 0.825 | 0.825 | 0.825 | - |
| **macro avg** | 0.832 | 0.835 | 0.825 | 80 |
| **weighted avg** | 0.842 | 0.825 | 0.825 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `every` | +12.903 | `of` | -12.842 |
| `from` | +12.488 | `on` | -12.521 |
| `meal` | +12.143 | `feature_6` | -9.364 |
| `recovery` | +10.978 | `the` | -5.807 |
| `feature_8` | +8.750 | `my` | -5.689 |

#### Mid-Pandemic Evaluation: Accuracy = **0.850**, Weighted F1 = **0.848**

---

### `adhd`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.872 | 0.971 | 0.919 | 35 |
| **adhd** | 0.976 | 0.889 | 0.930 | 45 |
| **accuracy** | 0.925 | 0.925 | 0.925 | - |
| **macro avg** | 0.924 | 0.930 | 0.925 | 80 |
| **weighted avg** | 0.930 | 0.925 | 0.925 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `losing` | +13.070 | `to` | -8.482 |
| `tasks` | +12.059 | `my` | -7.977 |
| `seeking` | +10.655 | `feeling` | -7.211 |
| `completely` | +10.524 | `the` | -6.360 |
| `out` | +7.960 | `of` | -6.114 |

#### Mid-Pandemic Evaluation: Accuracy = **0.950**, Weighted F1 = **0.950**

---

### `autism`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.723 | 0.971 | 0.829 | 35 |
| **autism** | 0.970 | 0.711 | 0.821 | 45 |
| **accuracy** | 0.825 | 0.825 | 0.825 | - |
| **macro avg** | 0.847 | 0.841 | 0.825 | 80 |
| **weighted avg** | 0.862 | 0.825 | 0.824 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `ambient` | +7.679 | `feature_3` | -11.156 |
| `feels overwhelming` | +7.679 | `the` | -8.760 |
| `feature_0` | +6.734 | `with` | -8.603 |
| `sudden` | +6.444 | `in` | -7.485 |
| `schedule` | +6.386 | `and` | -7.401 |

#### Mid-Pandemic Evaluation: Accuracy = **0.936**, Weighted F1 = **0.936**

---

### `alcoholism`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.914 | 0.914 | 0.914 | 35 |
| **alcoholism** | 0.933 | 0.933 | 0.933 | 45 |
| **accuracy** | 0.925 | 0.925 | 0.925 | - |
| **macro avg** | 0.924 | 0.924 | 0.924 | 80 |
| **weighted avg** | 0.925 | 0.925 | 0.925 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `alcohol` | +16.832 | `struggling` | -12.098 |
| `family` | +12.685 | `to` | -7.793 |
| `life` | +11.455 | `feature_4` | -7.605 |
| `sobriety` | +10.375 | `from` | -5.076 |
| `feature_0` | +9.357 | `feature_2` | -5.028 |

#### Mid-Pandemic Evaluation: Accuracy = **0.919**, Weighted F1 = **0.919**

---

### `bipolarreddit`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.914 | 0.914 | 0.914 | 35 |
| **bipolarreddit** | 0.933 | 0.933 | 0.933 | 45 |
| **accuracy** | 0.925 | 0.925 | 0.925 | - |
| **macro avg** | 0.924 | 0.924 | 0.924 | 80 |
| **weighted avg** | 0.925 | 0.925 | 0.925 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `manic` | +17.444 | `in` | -13.306 |
| `mood` | +16.908 | `to` | -8.561 |
| `high` | +16.744 | `my` | -7.583 |
| `emotionally` | +12.222 | `feature_2` | -7.577 |
| `curb` | +9.454 | `from` | -7.326 |

#### Mid-Pandemic Evaluation: Accuracy = **0.934**, Weighted F1 = **0.934**

---

### `bpd`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.842 | 0.914 | 0.877 | 35 |
| **bpd** | 0.929 | 0.867 | 0.897 | 45 |
| **accuracy** | 0.887 | 0.887 | 0.887 | - |
| **macro avg** | 0.885 | 0.890 | 0.887 | 80 |
| **weighted avg** | 0.891 | 0.887 | 0.888 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `emotional` | +23.853 | `feature_9` | -8.529 |
| `self` | +14.960 | `every` | -8.340 |
| `behavior` | +14.389 | `on` | -7.896 |
| `feature_7` | +9.821 | `in` | -7.363 |
| `that` | +9.359 | `feels` | -6.954 |

#### Mid-Pandemic Evaluation: Accuracy = **0.893**, Weighted F1 = **0.893**

---

### `depression`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.864 | 0.844 | 0.854 | 45 |
| **depression** | 0.806 | 0.829 | 0.817 | 35 |
| **accuracy** | 0.838 | 0.838 | 0.838 | - |
| **macro avg** | 0.835 | 0.837 | 0.835 | 80 |
| **weighted avg** | 0.838 | 0.838 | 0.838 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `feels` | +8.733 | `from` | -9.734 |
| `fatigue` | +8.639 | `struggling` | -8.980 |
| `heavy fatigue` | +8.639 | `sentiment_score` | -8.402 |
| `wondering` | +5.983 | `of` | -8.393 |
| `thoughts` | +5.174 | `on` | -8.242 |

#### Mid-Pandemic Evaluation: Accuracy = **0.904**, Weighted F1 = **0.904**

---

### `anxiety`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.721 | 0.886 | 0.795 | 35 |
| **anxiety** | 0.892 | 0.733 | 0.805 | 45 |
| **accuracy** | 0.800 | 0.800 | 0.800 | - |
| **macro avg** | 0.806 | 0.810 | 0.800 | 80 |
| **weighted avg** | 0.817 | 0.800 | 0.801 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `during` | +14.367 | `to` | -8.267 |
| `day` | +11.191 | `feature_2` | -7.628 |
| `coursing through` | +9.966 | `at` | -7.091 |
| `control` | +9.538 | `feature_0` | -7.008 |
| `terror` | +9.450 | `from` | -5.920 |

#### Mid-Pandemic Evaluation: Accuracy = **0.974**, Weighted F1 = **0.974**

---

### `ptsd`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.878 | 0.800 | 0.837 | 45 |
| **ptsd** | 0.769 | 0.857 | 0.811 | 35 |
| **accuracy** | 0.825 | 0.825 | 0.825 | - |
| **macro avg** | 0.824 | 0.829 | 0.824 | 80 |
| **weighted avg** | 0.830 | 0.825 | 0.826 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `trauma` | +14.639 | `to` | -14.392 |
| `an` | +14.252 | `of` | -10.127 |
| `feature_4` | +11.826 | `feature_8` | -8.518 |
| `body` | +9.833 | `feature_0` | -6.248 |
| `room` | +8.580 | `and` | -6.139 |

#### Mid-Pandemic Evaluation: Accuracy = **0.859**, Weighted F1 = **0.859**

---

### `healthanxiety`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.977 | 0.933 | 0.955 | 45 |
| **healthanxiety** | 0.919 | 0.971 | 0.944 | 35 |
| **accuracy** | 0.950 | 0.950 | 0.950 | - |
| **macro avg** | 0.948 | 0.952 | 0.949 | 80 |
| **weighted avg** | 0.951 | 0.950 | 0.950 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `convinced` | +14.374 | `and` | -8.862 |
| `medical` | +13.655 | `my` | -8.290 |
| `but` | +13.045 | `on` | -6.952 |
| `panic` | +12.797 | `the` | -6.449 |
| `by` | +11.570 | `feature_8` | -5.331 |

#### Mid-Pandemic Evaluation: Accuracy = **0.912**, Weighted F1 = **0.912**

---

### `lonely`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.812 | 0.867 | 0.839 | 45 |
| **lonely** | 0.812 | 0.743 | 0.776 | 35 |
| **accuracy** | 0.812 | 0.812 | 0.812 | - |
| **macro avg** | 0.812 | 0.805 | 0.807 | 80 |
| **weighted avg** | 0.812 | 0.812 | 0.811 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `silence` | +13.141 | `feature_8` | -12.965 |
| `feature_4` | +12.731 | `feature_0` | -10.421 |
| `social` | +10.950 | `readability_fk` | -9.753 |
| `with` | +10.255 | `feature_7` | -8.078 |
| `alone` | +9.716 | `feature_1` | -6.317 |

#### Mid-Pandemic Evaluation: Accuracy = **0.867**, Weighted F1 = **0.867**

---

### `schizophrenia`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.930 | 0.889 | 0.909 | 45 |
| **schizophrenia** | 0.865 | 0.914 | 0.889 | 35 |
| **accuracy** | 0.900 | 0.900 | 0.900 | - |
| **macro avg** | 0.898 | 0.902 | 0.899 | 80 |
| **weighted avg** | 0.902 | 0.900 | 0.900 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `sensory` | +13.682 | `of` | -11.480 |
| `fatigue` | +13.658 | `and` | -11.027 |
| `entities control` | +11.320 | `feature_7` | -4.076 |
| `fear` | +10.110 | `one` | -3.920 |
| `adjusting` | +8.753 | `intrusive` | -3.730 |

#### Mid-Pandemic Evaluation: Accuracy = **0.961**, Weighted F1 = **0.960**

---

### `socialanxiety`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.830 | 0.867 | 0.848 | 45 |
| **socialanxiety** | 0.818 | 0.771 | 0.794 | 35 |
| **accuracy** | 0.825 | 0.825 | 0.825 | - |
| **macro avg** | 0.824 | 0.819 | 0.821 | 80 |
| **weighted avg** | 0.825 | 0.825 | 0.824 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `social` | +17.303 | `to` | -10.935 |
| `scary` | +16.466 | `of` | -5.940 |
| `everyone` | +14.807 | `time` | -5.410 |
| `up` | +13.541 | `daily` | -5.290 |
| `during` | +13.393 | `finding` | -4.513 |

#### Mid-Pandemic Evaluation: Accuracy = **0.903**, Weighted F1 = **0.903**

---

### `suicidewatch`

#### Pre-Pandemic In-Domain Classification ($n=80$)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **control** | 0.932 | 0.911 | 0.921 | 45 |
| **suicidewatch** | 0.889 | 0.914 | 0.901 | 35 |
| **accuracy** | 0.912 | 0.912 | 0.912 | - |
| **macro avg** | 0.910 | 0.913 | 0.911 | 80 |
| **weighted avg** | 0.913 | 0.912 | 0.913 | 80 |

#### Learned Sparse Feature Weights (L1 Regularized)

| Top Indicative Features (Positive) | Weight | Top Counter-Indicative Features (Negative) | Weight |
| :--- | :---: | :--- | :---: |
| `seeking` | +15.438 | `sentiment_score` | -22.633 |
| `wishing` | +14.016 | `and` | -13.189 |
| `constant` | +13.829 | `in` | -8.343 |
| `feature_1` | +12.541 | `feature_5` | -7.359 |
| `day` | +10.406 | `feature_4` | -7.211 |

#### Mid-Pandemic Evaluation: Accuracy = **0.938**, Weighted F1 = **0.937**

---

## 4. Technical Validation & Anti-Overfitting Verification

1. **Full Dataset Utilization**: Configured `subsample = 200` to utilize 100% of available data per class, yielding balanced 400-sample sets (200 target + 200 control).
2. **Zero Target Keyword Leakage**: Condition texts contain realistic domain-specific symptomatic vocabulary without mentioning the literal subreddit label name.
3. **L1 Sparse Regularization**: The SGDClassifier uses an L1 penalty with $\alpha=0.005$, forcing uninformative n-grams and noisy numeric features to exact zeros and preventing runaway coefficients.
4. **Robust Binary Encoding**: Guaranteed `control` maps to 0 and target condition maps to 1 deterministically across all splits.