Set your API key:
```
export GOODFIRE_API_KEY=sk-goodfire-your-key
```

Run the simulation:
```
source .venv/bin/activate && uv pip install pyyaml python-dotenv goodfire altair streamlit pandas && uv run driver.py
```

Run the dashboard:
```
uv run streamlit run dashboard.py
```
