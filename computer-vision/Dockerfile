FROM python:3.8

WORKDIR /workspace
ADD requirements_dalle.txt /workspace/
#ADD examples /workspace/examples/

RUN pip install -r requirements_dalle.txt
#RUN pip install --no-cache-dir -r requirements_dalle.txt

ADD dalle.py /workspace/

RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


# You expose the gradio app at port 8000
EXPOSE 8000

CMD uvicorn dalle:app --host 0.0.0.0