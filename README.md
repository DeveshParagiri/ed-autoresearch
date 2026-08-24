# ED-Fire

ED-Fire tests whether autoresearch can develop a better fire module for the [Ecosystem Demography model](https://gel.umd.edu/ed.php). Global fire models still struggle to reproduce where fire occurs, when it peaks, and how much land burns. The research loop proposes interpretable changes to the model's equations, fits their parameters, and tests each formulation against satellite observations.

The model predicts monthly burned area from climate, vegetation, land use, population, and lightning. GFED5 and ILAMB measure whether a proposed mechanism improves the global spatial pattern, seasonal cycle, magnitude, and performance across 14 regions. The aim is to discover physical formulations for fuel availability, fuel moisture, ignition, spread, and human influence that improve the observations without turning the model into a black box. More context is available on the [Exaforge Earth System Models project page](https://exaforgelabs.com/research/projects/earth-system-models/).

## Autoresearch context

The research loop works entirely inside this directory:

```text
autoresearch/
├── __pycache__/
│   └── model.cpython-313.pyc
├── inputs/
│   ├── climate.nc
│   ├── ed.nc
│   ├── lightning.nc
│   ├── luh2.nc
│   ├── population.nc
│   └── README.md
├── model.py
├── research.md
└── results.tsv
```
