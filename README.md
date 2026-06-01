# Instructions on how to launch the API
In the AML-BRATS directory, in the terminal, use the command:

    uv run main.py

If you do not have uv installed, then run:

```
pip install -r requirements.txt

python main.py
```

The API runs locally, open your browser at http://127.0.0.1:8000/docs.

To get the app restarted:

    uvicorn api:app --reload

### To get the demo running, in another terminal, at the same time with the API running:

    streamlit run streamlit.py