#!/bin/bash
docker build -t app-immunogenn . 
docker run --rm -it \
    -v $(pwd)/test/:/home/biolib/test/ \
    -v $(pwd)/src/:/home/biolib/src \
    -v $(pwd)/run.py:/home/biolib/run.py \
    -v $(pwd)/data_record/:/home/biolib/data_record/ \
    app-immunogenn /bin/bash

    

#-v $(pwd)/esm/:/root/.cache/torch/hub/checkpoints/ \
