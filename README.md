# AsteroidProxOps
The code and data within this repository are done as part of the NASA Space Technology Graduate Research Opportunities (NSTGRO) fellowship:
Autonomous Landing and Proximity Operation Technology for Poorly-Characterized Small Bodies

Code will regenerate the results from the conference paper / presentation given at Astrodynamics Specialist Conference 2026 titled "Remote Guidance for a Mothership-Daughtership Formation for Asteroid Exploration" (AAS 26-783)

## How to run:
```sh
conda create -n ENV_NAME python=3.9.13
conda activate ENV_NAME
pip install poetry OR conda install poetry
poetry install
```
Confirm environment by running the test scripts.
In the root directory, run:
```sh
pytest
```
## How to run a single MPC simulation:

Within the repository, update input/config.ini to have the desired initial and final conditions or keep it as is to generate conference paper figures. Run src/MPC.py.

Readme last updated: 2026-06-25 (June 25, 2026)


