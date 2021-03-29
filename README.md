[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# Logistic Regression Algorithm with Pre-Processing Methods

## Libraries
```python
from tkinter import filedialog
import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
```

## Prerequisites
```python
missing_formats = ["NA", "nan", "NAN", ""]
delimeter = "," 
```
**missing formats** could be extended depending of needings _i.e. "N.A.", "null", etc._

**delimeter** is mostly notaded by ";" or "," in .csv files.

## Usage
### 1.  Detecting Misssing Columns
``` python
        self.missing_col_list = [column for column in df.columns if df[column].isnull().any()]
```

### 2.  Simple Imputations

``` python
        num_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        cat_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
```
Numerical and categorical variables are automatically defined.
For numerical columns, 'mean' imputation was used. Other imputation techniques can also be handled.

For categorical ones, 'most frequent' was chosed. 

### 3.  Encodings
One-hot encoding or label encoding are available to be used in.

#### Label Encoding
``` python
        labelencoder = LabelEncoder()
        for col in cat_columns:
            self.df[col] = labelencoder.fit_transform(self.df[col])

        return self.df
```
#### One-hot Encoding
``` python
        # generate binary values using get_dummies
        for col in cat_columns:
            self.df = pd.concat([pd.get_dummies(self.df[col], prefix='Type', drop_first=True), self.df], axis=1).drop([col], axis=1)
        return self.df
```

### 4.  Train-Test Split
0.8 and 0.2 Train-Test Split Sizes pre-defined.

``` python
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### 5.  Logistic Regression

``` python
        logr = LogisticRegression()
        logr.fit(self.X_train, self.y_train)
```

### 6. Metrics

``` python
        # Predicting the Test set results
        y_pred = logr.predict(self.X_test)

        # Confusion Matrix
        print(confusion_matrix(self.y_test, y_pred))

        # classification_report
        print(classification_report(self.y_test, y_pred))

        # score
        print(accuracy_score(self.y_test, y_pred))
```
## Example Result
              
              precision    recall  f1-score   support

           0        0.84      0.92      0.88       107
           1        0.76      0.62      0.68        47

    accuracy                            0.82       154
    macro avg       0.80      0.77      0.78       154
    weighted avg    0.82      0.82      0.82       154

**accuracy_score :** 0.8246753246753247 

## Contributing
Pull requests are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/orkunkus/logistic_regression.svg?style=for-the-badge
[contributors-url]: https://github.com/orkunkus/logistic_regression/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/orkunkus/logistic_regression.svg?style=for-the-badge
[forks-url]: https://github.com/orkunkus/logistic_regression/network/members
[stars-shield]: https://img.shields.io/github/stars/orkunkus/logistic_regression.svg?style=for-the-badge
[stars-url]: https://github.com/orkunkus/logistic_regression/stargazers
[issues-shield]: https://img.shields.io/github/issues/orkunkus/logistic_regression.svg?style=for-the-badge
[issues-url]: https://github.com/orkunkus/logistic_regression/issues
[license-shield]: https://img.shields.io/github/license/orkunkus/logistic_regression.svg?style=for-the-badge
[license-url]: https://github.com/orkunkus/logistic_regression/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/orkunkus


