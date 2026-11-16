# HomeWorth – House Price Predictor for Indian Cities

Enter the house you want (bedrooms, bathrooms, area, floor, age, facing, amenities …) and a machine-learning
model predicts its price. Built with **Python, scikit-learn and Streamlit** for 16 Indian cities and 150+ localities.

## Run it in VS Code (step by step)

1. Install **Python 3.10 – 3.12** (python.org, tick *"Add Python to PATH"*) and **VS Code** with the *Python* extension.
2. Unzip `HomeWorth.zip`. In VS Code choose **File → Open Folder…** and select the `HomeWorth` folder.
3. Open a terminal: **Terminal → New Terminal**.
4. Create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - macOS / Linux: `python3 -m venv venv` then `source venv/bin/activate`
5. Install the libraries (internet needed, one time): `pip install -r requirements.txt`
6. Start the app: `streamlit run app.py`
7. The browser opens at **http://localhost:8501**. Stop with `Ctrl + C`.

If VS Code asks for an interpreter: **Ctrl+Shift+P → Python: Select Interpreter** → choose the one inside `venv`.

> The first launch can take ~20 s: if your scikit-learn version differs from the one the bundled model was
> saved with, the app re-trains automatically (about 10 s).

## Before your demo

Edit the top of `src/config.py`:

```python
STUDENT_NAME = "Your Name"
COLLEGE_NAME = "Your College / University"
```

## What the app does

| Section | What it gives you |
|---|---|
| **Requirements form** | City, locality, property type, area, bedrooms, bathrooms, balconies, parking, floor, age, furnishing, facing, RERA, schools/hospitals nearby, modular kitchen, metro distance, amenities |
| **Predicted price** | Price in ₹ Cr / Lakh, 80% likely range, ₹ per sq ft, EMI, all-in buying cost, downloadable report |
| **Why this price?** | Which of your choices add or remove value; price vs. size curve |
| **Budget Finder** | Enter a budget → largest area you can afford, what each BHK gets you, which localities fit |
| **Compare Cities** | Your exact requirement priced in all 16 cities |
| **Market & Data** | City/locality rates, dataset exploration, correlation heatmap, price by BHK |
| **EMI & Buying Cost** | Loan calculator, repayment chart, stamp duty & costs |
| **My Predictions** | History of every prediction in the session (CSV download) |
| **About the Model** | Accuracy, model comparison, feature importance, error analysis |

## Project structure

```
app.py                  Streamlit interface
requirements.txt        Libraries
.streamlit/config.toml  Colour theme
data/housing_data.csv   28,000-record training data
models/                 Trained model + metrics
assets/                 Hero banner, 16 city scenes, 5 property illustrations, logo
src/
  config.py             Cities, localities, benchmark rates, options   <- edit rates here
  data_generator.py     Builds the dataset using Indian pricing logic
  train_model.py        Trains & compares 3 algorithms, saves metrics
  predictor.py          Loads model, predicts, what-if and budget logic
  finance.py            EMI, amortisation, stamp duty
  charts.py             All charts
  make_images.py        Draws every illustration (Pillow)
```

Useful commands (run from the project folder):

```
python -m src.data_generator   # rebuild dataset
python -m src.train_model      # re-train (about 15 s)
python -m src.make_images      # redraw illustrations
```

## How it works (for your viva)

1. **Data** – price = locality benchmark (₹/sq ft) × area × property type × age × floor × furnishing × facing ×
   amenities × metro distance × RERA …, plus ~9% random market noise.
2. **Preprocessing** – categorical columns one-hot encoded; price modelled on a log scale.
3. **Model** – `HistGradientBoostingRegressor` with monotonic constraints (more area never lowers the price).
   Compared with Ridge regression and Random Forest on a held-out 20% test set.
4. **Result** – R² ≈ 0.97, median error ≈ 7%. The data has ~9% built-in noise, so no model can honestly beat that.
5. **80% range** – taken from the model's real errors on unseen data.
6. **Budget Finder** – the model is monotonic in area, so a binary search finds the biggest area that fits a budget.

## Be upfront about the data

Training data is **simulated** from published market benchmarks (e.g. Anarock Q3 2026: top-7-city average
≈ ₹9,700 / sq ft). It is not scraped from listing portals, and locality rates are approximate. The *About* tab says so.
To use real data, replace `data/housing_data.csv` (same columns) and run `python -m src.train_model`.

*For education only – not a certified valuation.*
