FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install jupytext

COPY . .

RUN jupytext --to py main_copy.ipynb 

CMD ["python", "main.py"]