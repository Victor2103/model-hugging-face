FROM python:3.8

WORKDIR /workspace
ADD requirements.txt /workspace/
#ADD examples /workspace/examples/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ADD app.py /workspace/

RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


# You expose the gradio app at port 8000
EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0