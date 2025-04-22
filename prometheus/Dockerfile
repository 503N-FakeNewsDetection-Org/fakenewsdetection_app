# Use the official Prometheus base image
FROM prom/prometheus:v2.52.0

COPY prometheus.yml /etc/prometheus/prometheus.yml
COPY alert-rules.yml /etc/prometheus/alert-rules.yml

EXPOSE 9090

