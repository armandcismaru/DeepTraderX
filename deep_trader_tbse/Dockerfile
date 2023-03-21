FROM python:3.9

WORKDIR /app

RUN pip install pandas
RUN pip install numpy
RUN pip install pylint
RUN pip install boto3
RUN pip install keras
RUN pip install tensorflow
RUN pip install progress

COPY . /app

CMD ["python3", "tbse.py", "markets.csv"]

EXPOSE 80/tcp
EXPOSE 80/udp
