FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORkDIR /root

ADD /app /root/app
ADD /artifacts /root/artifacts
ADD /requirements.txt /root/requirements

RUN pip3 install -r requirements

EXPOSE 8000

WORKDIR /root/app

ENV DEVICE="cuda"

CMD python3 -m uvicorn main:app --host=0.0.0.0 --port=8000