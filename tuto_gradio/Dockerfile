FROM python:3.8

WORKDIR /workspace

# Download the python requirements
COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# COPY the fast api app
COPY app.py /workspace/

# Run as 420 user
RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


# You expose the fast api app at port 8000
EXPOSE 7860

CMD python3 app.py