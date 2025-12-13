FROM apache/airflow:2.9.3

USER root

COPY airflow/requirements.txt /tmp/airflow-requirements.txt
RUN chown airflow:root /tmp/airflow-requirements.txt

USER airflow

RUN pip install --no-cache-dir -r /tmp/airflow-requirements.txt \
    && rm -f /tmp/airflow-requirements.txt
