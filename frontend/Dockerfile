FROM ubuntu

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /.venv
ENV PATH="/.venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]