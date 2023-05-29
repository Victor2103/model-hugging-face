FROM python:3.8

WORKDIR /workspace
ADD requirements_fastApi.txt test.py /workspace/

RUN pip install --no-cache-dir -r requirements_fastApi.txt

RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


# You expose the fast api app at port 8000
EXPOSE 8000

CMD uvicorn test:app