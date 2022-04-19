FROM python:3.8-slim
LABEL maintainer="u6k.apps@gmail.com"

# Install
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        graphviz \
        graphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -U pip && \
    pip install pipenv

# Install python packages
WORKDIR /var/myapp
VOLUME /var/myapp

COPY Pipfile Pipfile.lock ./
RUN pipenv sync --dev

VOLUME /var/output

CMD ["pipenv", "scripts"]
