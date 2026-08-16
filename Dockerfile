FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY apps ./apps
COPY career_model.pkl ./
COPY feature_columns.pkl ./
COPY questions.json ./

EXPOSE 8000

CMD ["uvicorn", "apps.main:app", "--host", "0.0.0.0", "--port", "8000"]