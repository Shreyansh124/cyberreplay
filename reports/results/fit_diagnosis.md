| model              |   fit_rows |   val_rows |   train_f1_macro |   val_f1_macro |    gap | diagnosis               |
|:-------------------|-----------:|-----------:|-----------------:|---------------:|-------:|:------------------------|
| LogisticRegression |    1407859 |     351965 |           0.7204 |         0.7164 | 0.0041 | underfit (high bias)    |
| DecisionTree       |    1407859 |     351965 |           0.9733 |         0.9181 | 0.0553 | overfit (high variance) |
| RandomForest       |    1407859 |     351965 |           0.9651 |         0.9477 | 0.0173 | good fit                |
| XGBoost            |    1407859 |     351965 |           0.9706 |         0.9441 | 0.0264 | slight overfit          |
| Dummy              |    1407859 |     351965 |           0.1146 |         0.1146 | 0.0000 | underfit (high bias)    |
