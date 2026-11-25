# Model analysis and compatibility

This document records what was verified about `models/boston_home_price_prediction.h5`
before the app was built, and how each conclusion was reached.

## 1. File and framework

| Item | Value | How verified |
|---|---|---|
| Format | HDF5 (legacy Keras `.h5`) | `h5py` attributes |
| Saved with | Keras 3.15.1, TensorFlow backend | `keras_version`, `backend` attributes |
| Loader | `keras.models.load_model(path, compile=False)` | loaded successfully with TF 2.21 / Keras 3.15.1 |
| Architecture | `Sequential`: Dense 64 relu → Dense 32 relu → Dense 16 relu → Dense 1 (linear) | `model_config` + `model.summary()` |
| Parameters | 3,521 (≈14 KB) | `model.summary()` |
| Training config | loss `mse`, metric `mae`, Adam lr 0.001 | `training_config` attribute |
| SHA-256 | recorded in `models/preprocessing.json` → `model_sha256` | checked by tests on every run |

`compile=False` is used because the app only runs inference. The file is never written to.

## 2. Input

* Shape `(None, 13)`: a single flat feature vector per sample, batch dimension first.
* dtype `float32`.
* The app always sends exactly one row: `(1, 13)`.

## 3. Training context (where the preprocessing came from)

The `.h5` file has no preprocessing information. The training notebook was found at
`E:\jupyter\Learning\DL\Chapter-8 (ANN for Regression)\Docs\Chapter 8.1 ANN For Regression.ipynb`.
The `.h5` saved next to it has the same MD5 as the file in this project
(`43022500d0c3a3a704fd1ebe10543481`).

The notebook does:

```python
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()   # keras.datasets
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.fit_transform(X_test)      # re-fits the scaler on test data
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)
...
new_data = scaler.transform(new_data)      # uses the TEST-fitted scaler
model.save('boston_home_price_prediction.h5')
```

### Conclusions

* **Features:** the 13 columns of `keras.datasets.boston_housing`, in its fixed order:
  `CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT`.
* **Scaling:** `StandardScaler` (`(x − mean) / std`, population std). The model's
  weights were learned on **training-split** statistics.
* **Scaler persistence:** the fitted scaler was **not** saved. `boston_housing.load_data()`
  is deterministic (`seed=113`, `test_split=0.2`), so the exact training-split mean and
  std can be recomputed. `scripts/export_preprocessing.py` does this and writes
  `models/preprocessing.json`. The values match `sklearn.preprocessing.StandardScaler`
  to 1e-10 (this is tested).
* **No other transforms:** there is no encoding, log transform or missing-value handling.
  `CHAS` is already 0/1 and `RAD` is an integer index.

### Provenance check

The notebook printed `Predicted House Price: [[29.239614]]` for the sample row
`[0.02, 18, 2.5, 0, 0.45, 6.5, 65, 4.5, 1, 300, 15, 390, 5]` using its test-fitted scaler.
Repeating that exact computation with this project's model file gives **29.239614**. This
confirms that both the model file and the reconstructed dataset split are the ones from the notebook.

### Notebook scaler bug and how the app handles it

The notebook's prediction cell used a scaler re-fitted on `X_test`, which is not the scaling
the network was trained with. The app uses the **training-split** scaler, which is
correct for this network:

| Scaling used at inference | Same sample row | MAE on 102-row test split |
|---|---|---|
| Training-split scaler (**app**) | 28.0736 | **2.82** |
| Test-split scaler (notebook cell) | 29.2396 | 2.93 |

## 4. Output

* Shape `(None, 1)`, linear activation (unbounded regression).
* Target: `y` from `keras.datasets.boston_housing`, which is **MEDV**, the median value of
  owner-occupied homes in a census tract, **in thousands of US dollars (1970s)**. Training
  targets range from 5.0 to 50.0 (median 20.75). Values were capped at 50.0 in the original data.
* Display: `predicted_value × 1000` formatted as USD (e.g. 28.0736 → `$28,074`), always shown
  with the raw output and the "1970s, neighbourhood median" context.
* Negative outputs are possible for extreme inputs. They are never clamped. The UI shows the raw
  value and marks the dollar figure as not meaningful. Outputs above 50 come with a warning about the cap.

## 5. Input validation policy

* **Hard limits** (rejected with 422) are physical or definitional, e.g. percentages 0–100,
  `CHAS ∈ {0,1}`, `RAD` integer 1–24, `B ≤ 396.9` (the maximum of `1000(Bk − 0.63)²`).
* **Training range** (accepted, with a warning): values the model never saw. The prediction is
  an extrapolation there.
* **No silent defaults:** every feature is required. The "Use training medians" button fills
  the training-data medians only when the user asks for it, and says so.

## 6. Ethical limitations

`B` is derived from the proportion of Black residents, and `LSTAT` is a 1970 "lower status"
measure. Both are required inputs: removing them or fixing them to a constant would change the
model's input distribution and make its outputs meaningless. The app therefore keeps them, puts
them in a clearly labelled **Historical Dataset Variables** group, and shows a disclaimer.
The app is an educational demonstration and must not be used for valuation, lending, housing or
investment decisions.
