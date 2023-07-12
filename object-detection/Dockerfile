FROM python:3.8

WORKDIR /workspace

# Download the python requirements
COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# COPY the fast api app
COPY app.py /workspace/

# COPY the examples images
COPY examples /workspace/examples/

# Run as 420 user
RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


# You expose the fast api app at port 8000
EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port 8000