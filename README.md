# Forecasting MCC & Regular Payments

Comprehensive project for transaction forecasting using statistical, ML and DL approaches.

## Task Overview

The project solves two forecasting tasks:

### MCC Aggregates Forecasting

**Goal:** Forecast aggregated transaction amounts by categories at client level.

- **Forecasting horizons:**
  - Week +1 (weekly forecast)

- **Key features:**
  - Data aggregation by `client_id × category`
  - Series filtering: length ≥ 104 weeks, non-zero ratio ≥ 30%
  - Bottom-up vs Direct comparison for monthly forecast



## MCC Category Mapping

The project uses 16 simplified categories instead of raw MCC codes for better forecasting performance:

1. **food** - Food & Dining (restaurants, grocery stores, fast food)
2. **transport** - Transportation (gas stations, tolls, taxis, airlines)
3. **utilities** - Utilities & Bills (electric, gas, telecom, postal)
4. **retail** - Retail & Shopping (department stores, clothing, electronics)
5. **health** - Healthcare (doctors, dentists, hospitals, pharmacies)
6. **entertainment** - Entertainment (movies, amusement parks, digital goods)
7. **services** - Professional Services (beauty, cleaning, legal, accounting)
8. **home** - Home & Garden (hardware, lumber, garden supplies)
9. **education** - Education & Books (bookstores, art supplies)
10. **financial** - Financial Services (money transfer)
11. **tech** - Technology (computer services, equipment)
12. **hospitality** - Travel & Lodging (hotels, motels)
13. **sports** - Sports & Recreation (sporting goods, sports apparel)
14. **specialty** - Specialty Stores (pharmacies, antiques, florists)
15. **industrial** - Industrial & Manufacturing (metal work, tools, manufacturing)
16. **freight** - Freight & Logistics (trucking, freight)
17. **automotive** - Automotive (auto parts and accessories)

*Mapping details available in: `data/mcc_mapping.json`*

## Available Models

###  MCC Aggregates Forecasting

| Category     | Models                                          | Status |
|---------------|-------------------------------------------------|--------|
| Baselines     | Seasonal Naïve, Random Walk, ETS                |      |
| Statistics    | SARIMA, Prophet                      |      |
| ML, DL     | CatBoost, LSTM, Temporal Fusion Transformer (TFT)          |      |


## Metrics

### MCC Aggregates Forecasting

| Horizon        | Main Metrics        | Description                           |
|-----------------|-------------------------|-------------------------------------|
| Week +1         | sMAPE_w, RMSSE_w        | Weekly forecast errors                |

**Per-client benefit indicator:** Δ_u = sMAPE_m_BU - sMAPE_m_D


## Validation and Testing

### Validation Methodology

- **Task A:** Expanding CV with 80% train / 10% val / 10% test per-series


### Feature Engineering

#### MCC Aggregates Forecasting
- **Transformations:** log1p → z-score normalization
- **Calendar features:** day of week, month, quarter, holidays (ES)
- **Lags:** 1, 2, 3, 4, 8, 12, 52 weeks
- **Moving averages:** SMA with windows 4, 12 weeks

