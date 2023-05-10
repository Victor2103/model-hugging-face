FROM python:3.8

WORKDIR /workspace
ADD requirements_2.txt bastien_model.py /workspace/

RUN pip install --no-cache-dir -r requirements_2.txt

RUN apt-get -y update

RUN apt-get install -y ffmpeg

RUN chown -R 42420:42420 /workspace/
ENV HOME=/workspace/


#If you deploy the chatbot you expose at port 5005.
EXPOSE 7860

CMD python3 bastien_model.py