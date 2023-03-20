#!/bin/bash

python configure_cloud.py

kubectl apply -f market-simulations.yaml
kubectl apply -f market-simulations-service.yaml

python deep_trader_tbse/deep_trader/utils.py
