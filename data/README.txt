DATASETS
========

1. BRFSS 2023 (Behavioral Risk Factor Surveillance System)
   File: LLCP2023.XPT
   Source: CDC — https://www.cdc.gov/brfss/annual_data/annual_2023.html
   Format: SAS Transport (.XPT), loaded via pyreadstat
   Size: ~433k rows, 350 columns
   Used for: Self-reported behavioral health experiment (asymmetric noise)

2. UCI Breast Cancer Wisconsin
   File: N/A — loaded directly from sklearn.datasets.load_breast_cancer()
   Source: UCI ML Repository via scikit-learn
   Size: 569 samples, 30 features
   Used for: Standard clinical dataset comparison (symmetric noise)
