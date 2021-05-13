# AUTHOR: Team Phoenix
# DESCRIPTION: driver-drowsiness container
# BUILD: docker build -t driver-drowsiness:latest .

FROM python:3.6-slim-stretch

LABEL maintainer="TeamPhoenix_"

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

# Airflow variables
ARG AIRFLOW_VERSION=1.10.10
ARG AIRFLOW_USER_HOME=/usr/local/airflow
ARG AIRFLOW_DEPS=""
ARG PYTHON_DEPS=""
ENV AIRFLOW_HOME=${AIRFLOW_USER_HOME}

# Define en_US
ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8

RUN set -ex \
    && buildDeps=' \
        freetds-dev \
        libkrb5-dev \
        libsasl2-dev \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        git \
    ' \
    && apt-get update -yqq \
    && apt-get upgrade -yqq \
    && apt-get install -yqq --no-install-recommends \
        $buildDeps \
        freetds-bin \
        build-essential \
        default-libmysqlclient-dev \
        apt-utils \
        curl \
        rsync \
        netcat \
        locales \
    && sed -i 's/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g' /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && useradd -ms /bin/bash -d ${AIRFLOW_USER_HOME} airflow \
    && pip install -U pip setuptools wheel \
    && pip install pytz \
    && pip install pyOpenSSL \
    && pip install ndg-httpsclient \
    && pip install pyasn1 \
    && pip install apache-airflow[crypto,celery,postgres,hive,jdbc,mysql,ssh${AIRFLOW_DEPS:+,}${AIRFLOW_DEPS}]==${AIRFLOW_VERSION} \
    && pip install 'redis==3.2' \
    && pip install 'tensorflow' \
    && pip install 'opencv-python' \
    && pip install keras \
    && if [ -n "${PYTHON_DEPS}" ]; then pip install ${PYTHON_DEPS}; fi \
    && apt-get purge --auto-remove -yqq $buildDeps \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y git \
    && apt-get install ffmpeg libsm6 libxext6 libglib2.0-0 -y

COPY config/airflow.cfg ${AIRFLOW_USER_HOME}/airflow.cfg
COPY requirements.txt requirements.txt
RUN pip install git+https://github.com/fastai/fastai.git
RUN pip install -r requirements.txt
COPY data ${AIRFLOW_USER_HOME}/data
COPY dags ${AIRFLOW_USER_HOME}/dags
COPY scripts ${AIRFLOW_USER_HOME}/scripts
COPY static ${AIRFLOW_USER_HOME}/scripts/static
COPY templates ${AIRFLOW_USER_HOME}/scripts/templates
COPY app.py ${AIRFLOW_USER_HOME}/scripts/app.py
COPY weights_best_vgg16_model2.hdf5 ${AIRFLOW_USER_HOME}/scripts/weights_best_vgg16_model2.hdf5

# Install required libraries
EXPOSE 8008 8080
#USER airflow
#RUN chown -R airflow:airflow ${AIRFLOW_USER_HOME}
WORKDIR ${AIRFLOW_USER_HOME}
COPY scripts/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["webserver"]
