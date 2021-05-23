#!/bin/bash

# reg=0.01, 0.05, 0.1, 0.5, 1.0, 5, 10, 50, 100
bash exp_batch.sh cora CE 200 100 0
bash exp_batch.sh cora CW 0.1 200 0

bash exp_batch.sh citeseer CE 200 100 0
bash exp_batch.sh citeseer CW 0.1 200 0

bash exp_batch.sh blogcatalog CE 200 100 0
bash exp_batch.sh blogcatalog CW 0.1 200 0

bash exp_batch.sh polblogs CE 200 100 0
bash exp_batch.sh polblogs CW 0.1 200 0