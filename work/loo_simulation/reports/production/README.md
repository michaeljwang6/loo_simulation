# Production reporting bundle

This directory contains paper-ready tables and figures generated from the
complete 100-replication Monte Carlo result.

Regenerate the bundle from the repository root with:

```powershell
& .\.venv311\Scripts\python.exe -m pip install -e ".[report]"
$env:MPLCONFIGDIR = (Resolve-Path .).Path + "\.mplconfig"
& .\.venv311\Scripts\python.exe scripts\report_results.py `
  --input results\full_ladder_production\merged `
  --output reports\production
```

The PNG files are convenient for review and slides. The matching PDF files are
vector outputs for papers. The LaTeX fragments use `booktabs`; include
`\usepackage{booktabs}` in the paper preamble.

The unconditional results remain primary. Any figure or table labeled
stable-only conditions on the fit's stability diagnostic and must be shown
together with the corresponding instability rate.

The project procedure is the **low-rank plug-in without LOO correction**.
