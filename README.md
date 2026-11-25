# Dose-dependent AEC and ATP kinetics simulator

This project simulates radiation dose-dependent adenylate energy charge (AEC)
using a Monte Carlo cell-damage model and an ODE model of ATP, ADP, and AMP
kinetics. It includes a Tkinter interface with live progress, measurements,
and plots shown after the simulation finishes.

## Requirements

- Python 3.10 or later
- Tk support in the Python installation

Install the scientific dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the interactive application

```powershell
python "Doz bağımlı AEC Eğrileri Çıktıları.py"
```

The application starts a simulation with the default settings. Change the
settings on the right and select **Simulasyonu Calistir** to run another one.
The results table and three plots update when the run completes.

By default, the CSV, Excel, and PNG outputs are saved in the `ciktilar` folder
next to the Python script. The application displays the full output path when
the run completes. Use `--output-dir` to choose another folder.

## Command-line options

```powershell
python "Doz bağımlı AEC Eğrileri Çıktıları.py" `
  --dose-min 0 --dose-max 5 --dose-points 26 `
  --priming-dose 0.05 --n-cells 3000 --seed 42 `
  --uncertainty-replicates 5 --output-dir "C:\AEC_CIKTILAR"
```

For a non-interactive run, add `--headless --no-show`.

## Outputs

- `simulasyon_ciktilari.csv` — dose response, AEC uncertainty, and adenylate
  fractions.
- `simulasyon_ciktilari.xlsx` — the same results in separate, formatted
  spreadsheet columns.
- `doz_bagimli_aec_egrileri.png` — combined overview of the three plots.
- `aec_doz_yaniti.png` — detailed dose-response plot (A).
- `aec_zaman_kinetigi.png` — detailed time-course plot (B).
- `adenilat_mol_kesirleri.png` — detailed ATP, ADP, and AMP fraction plot (C).

## Model note

This is a computational model, not clinical or medical advice. Interpret its
parameters and results in the context of the model assumptions and validate
them against suitable experimental data before scientific or clinical use.
