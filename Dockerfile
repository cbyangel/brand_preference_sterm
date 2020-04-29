FROM ***.***.com/kube-ops/docker-images/all-spark-with-koalas:latest
USER root
ENV YEONDYS_USER_HOME /home/yeondys

COPY requirements.txt .
RUN pip install -r ./requirements.txt
RUN useradd -ms /bin/bash -d ${YEONDYS_USER_HOME} yeondys
WORKDIR ${YEONDYS_USER_HOME}

USER yeondys
COPY yeondys.py .
COPY yeondys ./yeondys

ENTRYPOINT ["/usr/bin/python", "yeondys.py"]