#!/bin/bash

kubectl apply -f market-simulations.yaml
kubectl apply -f market-simulations-service.yaml

# Pickle and normalise the data in the S3 buckets
python deep_trader_tbse/src/deep_trader/utils.py
