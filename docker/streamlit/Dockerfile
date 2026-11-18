FROM python:3.12-slim

RUN pip install --no-cache-dir "streamlit>=1.40" "httpx>=0.27"

WORKDIR /app
COPY .streamlit ./.streamlit
COPY streamlit_app ./streamlit_app

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
