
# About

This folder contains prepared data of workflow executions.

The execution logs were first collected using [HuperFlow](https://github.com/hyperflow-wms/hyperflow),
and then parsede using [Log-parser](https://github.com/hyperflow-wms/log-parser).
This data was then converted to OpenMetric format and ingested in to prometheus using provided [Converter](../notebooks/metrics_ingestion.ipynb)

# Starting Prometheus server

To start the Prometheus with the data collected from workflow execution,
First things first unzip the `prometheus-db.zip` archive
```bash
unzip prometheus-db.zip
```
and then start the Prometheus specifying the path to the extracted folder
```bash
docker run -d -p 9090:9090 \
    -v prometheus-db:/prometheus \
    --name prometheus \
    prom/prometheus \
    --storage.tsdb.retention.time=3y \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --storage.tsdb.allow-overlapping-blocks \
    --query.lookback-delta=3s \
    --web.console.libraries=/usr/share/prometheus/console_libraries \
    --web.console.templates=/usr/share/prometheus/consoles
```
