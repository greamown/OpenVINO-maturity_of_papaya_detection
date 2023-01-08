FROM openvino/ubuntu18_dev:latest
# ------------------------------------------------------------------
WORKDIR /workspace
COPY ["./format_print.sh", "./requirements.sh", "/workspace/" ]
USER root
# ------------------------------------------------------------------
ENV DEBIAN_FRONTEND noninteractive  
RUN chmod +x ./requirements.sh \
&& ./requirements.sh \
&& rm ./*.sh
# ------------------------------------------------------------------
# 寫入進入W
ENTRYPOINT [ "/bin/bash", "-c" ]
