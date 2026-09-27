# Lab 1.1 — Data Exploration for Purchase Prediction

## Purpose
This lab performs an initial exploratory analysis of the Retailrocket event data as part of a gradient boosting purchase prediction workflow. The goal is to understand user behavior, event distribution, and dataset quality before feature engineering and model building.

## Setup
- Python 3.11+
- Required packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
- Input data files expected under `data/`:
  - `events.csv`
  - `item_properties_part1.csv`
  - `item_properties_part2.csv`

## How to Run
From the lab directory, run:

```bash
python data_exploration.py
```

The script reads the CSV files, performs summary analysis, and generates a visualization.

## Outputs
The script prints:
- dataset shapes
- event type counts and percentages
- missing value counts
- timestamp range
- user/item summary statistics

It also creates:
- `output/01_eda_overview.png`

## Key Design Choices
- Combined the two item property files vertically with `pd.concat(..., ignore_index=True)`.
- Converted millisecond timestamps to readable datetimes using `pd.to_datetime(..., unit='ms')`.
- Used `visitorid` grouping to measure user activity.
- Used a log-scaled y-axis for the events-per-user histogram to better show skewed behavior.
- Added bar labels to improve readability of the event distribution chart.

## Key Findings
Typical findings from this exploration include:
- The dataset contains multiple event types with uneven distribution.
- User activity is highly skewed, with a small number of users generating many events.
- The event timeline spans a clear date range after timestamp conversion.
- The data is suitable for downstream feature engineering after basic inspection.

## Extra Info
- The script suppresses warnings for cleaner notebook/script output.
- The plot is saved at 150 DPI for report-quality use.
- If the `output/` folder does not exist, it should be created before running the script.
