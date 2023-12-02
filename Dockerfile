FROM continuumio/miniconda3 AS build

COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda install nomkl && \
    conda install -c conda-forge conda-pack && \
    conda clean -a
RUN find /opt/conda/ -follow -type f -name '*.pyc' -delete

ENTRYPOINT ["/bin/bash"]
RUN conda-pack -n dpr_demo -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar

RUN /venv/bin/conda-unpack

# Runtime stage:
FROM debian:buster AS runtime
# Copy /venv from the previous stage:
COPY --from=build /venv /venv
ADD . .
SHELL ["/bin/bash", "-c"]
EXPOSE 5000

ENTRYPOINT source /venv/bin/activate && python demo.py