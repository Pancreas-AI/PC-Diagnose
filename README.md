# PC-Diagnose

Source code for a pancreatic cancer auxiliary diagnosis and analysis project.

This repository contains reproducible analysis scripts, a stripped Jupyter
notebook, and the Python/Tkinter source code for the desktop diagnostic tool.
It intentionally does not include patient data, generated reports, figures,
Excel outputs, packaged macOS apps, or zip archives.

## Contents

- `analysis/eus_model/`
  - Main EUS-FNA/B first-puncture undiagnosed classifier pipeline.
  - Reproducible notebook skeleton with outputs cleared.
- `analysis/cytology_regression/`
  - Logistic regression scripts for cytology positivity analyses.
  - Includes the latest specified two-model analysis where cytology result
    `6=malignant` is treated as positive and results `1-5` are treated as not
    reaching positivity.
- `app_source/`
  - Source code for the Chinese macOS desktop auxiliary diagnosis app.
  - This is source code only, not a packaged `.app`.
- `scripts/`
  - Local macOS helper scripts for launching and packaging during development.

## Data Policy

Clinical source workbooks, cleaned patient-level CSV files, generated tables,
figures, PDFs, PPT files, and packaged apps are excluded from this repository.

To reproduce analyses locally, place the required source workbooks in the
project root with the filenames expected by each script, for example:

- `工作簿1.xlsx`
- `2026胰腺癌数据汇总.xlsx`

Do not commit patient-level data to this repository.

## Environment

Python 3.10+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

The desktop app uses `tkinter`, which is bundled with most Python distributions
on macOS. Packaging the app requires PyInstaller.

## Run Examples

EUS diagnosis model analysis:

```bash
python analysis/eus_model/analysis_pipeline.py
```

Cytology regression, latest specified models:

```bash
python analysis/cytology_regression/analysis_specified_models_result6.py
```

Desktop app source:

```bash
python app_source/diagnosis_app.py
```

The app source expects local training/performance data files in the paths used
by `app_source/diagnosis_model.py`. Those files are intentionally not tracked in
Git.

## Medical Disclaimer

This code is for research and clinical decision-support exploration only. It is
not a substitute for pathology, multidisciplinary clinical judgment, or formal
medical device validation.
