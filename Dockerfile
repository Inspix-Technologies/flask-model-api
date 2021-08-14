FROM python:3.8.11-buster

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install Flask
CMD python app.py --port 8001