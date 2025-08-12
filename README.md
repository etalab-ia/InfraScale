# InfraScale

Infrascale is a tool developed by the Albert API team to estimate GPU requirements to do model inference at scale.

## Quickstart

- Create and activate a python virtual environment
- Install required dependencies with `pip install -r requirements.txt`
- Run with `streamlit run app.py`

## Contributing

The list of available GPUs and models in Infrascale can be found in `db/gpu.json` and `db/models.json`. New GPUs/models can be added by appending them directly at the bottom of the json files and following the current data format.

Locales are set in the `translations` folder. When adding a new language, don't forget to update the LANGUAGES dictionnary in `language.py`.

The app logic is entirely set in `solver.py`. Notebooks `solver.ipynb` and `calibration/calibration.ipynb` may be useful to understand the logic and how it was defined. A comprehensive article is currently being written to further explain how Infrascale works. 