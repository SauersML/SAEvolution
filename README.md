## Quickstart

Set your API key:
```
export GOODFIRE_API_KEY=sk-goodfire-your-key
```

Run the simulation:
```
source .venv/bin/activate && uv pip install pyyaml streamlit python-dotenv goodfire altair streamlit pandas && uv run driver.py
```

Run the dashboard:
```
uv run streamlit run dashboard.py
```

Resume run:
```
uv run driver.py --resume-latest
```

**Command-line arguments (driver)**

*   `--config-file <path>`: Specify a path to a configuration file (default: `config.yaml`).
*   `--resume-run-id <run_id>`: Resume a specific simulation run by providing its ID (found in the `simulation_state` directory).
*   `--resume-latest`: Automatically resume the most recently modified simulation run found in the `simulation_state` directory.
