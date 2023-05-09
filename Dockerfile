FROM python:3.8

WORKDIR /workspace
ADD requirements.txt app.py /workspace/

RUN pip install --no-cache-dir -r requirements.txt


RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


#If you deploy the chatbot you expose at port 5005.
EXPOSE 7860

CMD python3 app.py