# Predikcija čvrstoće betona

Projekat predviđa pritisnu čvrstoću betona (MPa) na osnovu sastava mešavine i starosti, koristeći tri modela: OLS linearnu regresiju, Random Forest i XGBoost.

## Istraživačka pitanja

1. Koji sastojci mešavine najviše utiču na čvrstoću?
2. Da li je linearna regresija dovoljna, ili su neophodni nelinearni modeli?
3. Koji sastav i starost betona daju optimalnu predviđenu čvrstoću?

## Podaci

Dataset sadrži **1.030 uzoraka** sa 8 ulaznih karakteristika i ciljnom promenljivom:

| Karakteristika | Opis |
|---|---|
| Cement | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash | kg/m³ |
| Water | kg/m³ |
| Superplasticizer | kg/m³ |
| Coarse Aggregate | kg/m³ |
| Fine Aggregate | kg/m³ |
| Age | dani (1–365) |
| **Strength** | **MPa — ciljna promenljiva** |

Inženjering karakteristika dodaje `water_cement_ratio = Water / Cement` (Abrams-ov zakon).

## Rezultati modela (test skup)

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| OLS | 9.37 | 7.14 | 24.21% |
| Random Forest | 6.17 | 4.26 | 14.26% |
| **XGBoost** | **5.49** | **3.71** | **11.73%** |

XGBoost nadmašuje OLS za ~41% po RMSE. OLS pretpostavke su narušene (nelinearnost, nenormalnost reziduala), što opravdava korišćenje nelinearnih modela.

## Struktura projekta

```
├── data/
│   └── concrete_data.csv      # dataset
├── src/
│   ├── preprocessing.py       # učitavanje, feature engineering, podela skupa
│   ├── models.py              # treniranje OLS, RF, XGBoost (GridSearchCV)
│   ├── evaluation.py          # metrike (RMSE, MAE, MAPE) i pokretanje pipeline-a
│   ├── validation.py          # provera pretpostavki OLS modela
│   └── visualization.py       # generisanje grafika
├── figures/                   # generisani PNG grafici
└── notebook.ipynb             # kompletna analiza sa narativom
```

## Pokretanje

```bash
pip install -r requirements.txt

# Kompletan pipeline: treniranje sva tri modela + grafici
python src/evaluation.py

# OLS dijagnostika pretpostavki
python src/validation.py

# Samo generisanje grafika
python src/visualization.py

# Interaktivna analiza
jupyter notebook notebook.ipynb
```
