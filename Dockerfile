FROM r-base

COPY . /src

WORKDIR /src

RUN Rscript dependencies.R

RUN useradd -ms /bin/bash rags

RUN install -m 0755 libpardiso600-GNU720-X86-64.so /usr/local/lib/ \

RUN chown -R rags:rags .

USER rags