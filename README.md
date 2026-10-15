# NovaRetail Customer Intelligence Dashboard

## Files
- `app.py` — Streamlit application
- `NR_dataset.xlsx` — NovaRetail source dataset
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with GitHub + Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `NR_dataset.xlsx` to the repository root.
3. In Streamlit Community Cloud, create a new app from the repository and select `app.py` as the entry point.
4. Deploy.

The dashboard works without an external API for common grounded questions. To enable broader natural-language AI responses, add an OpenAI API key as a Streamlit secret:

```toml
OPENAI_API_KEY = "your-key-here"
```

Do **not** commit the API key to GitHub.

## Data interpretation notes
- The workbook contains 100 purchase records, although the project brief references 99 sampled consumers.
- One record has a missing segment label and is shown as `Unclassified` rather than being imputed.
- Repeated `CustomerID` values can carry different segment labels, so the app does not force a single inferred segment onto a customer.
- Repeated `TransactionID` values exist; the dashboard therefore treats each source row as a purchase record rather than assuming `TransactionID` is unique.
- Product category labels are preserved exactly as supplied instead of merging similar labels without a documented taxonomy.
