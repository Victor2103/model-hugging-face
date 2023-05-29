FROM python:3.8

WORKDIR /workspace
ADD requirements_textGeneration.txt textGeneration.py /workspace/

RUN pip install --no-cache-dir -r requirements_textGeneration.txt

RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


# You expose the fast API app at port 8000
EXPOSE 8000

CMD uvicorn textGeneration:app --host=0.0.0.0