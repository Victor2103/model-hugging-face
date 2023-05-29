FROM python:3.8

WORKDIR /workspace
ADD requirements_textGeneration.txt textGeneration.py /workspace/

RUN pip install --no-cache-dir -r requirements_textGeneration.txt

RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


# You expose the gradio app at port 7860
EXPOSE 7860

CMD python3 textGeneration.py