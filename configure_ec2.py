# pylint: skip-file
"""
Configure a cloud environment for the Kubernetes cluster.
"""

from datetime import datetime
import os
import logging
import sys
import boto3
from botocore.exceptions import ClientError

# Define the region where you want to create the EC2 instances
region = "us-west-2"

# Create an EC2 client object
ec2 = boto3.client("ec2", region_name=region)

# Set the AMI ID for the instance
ami_id = "ami-0557a15b87f6559cf"

# Set the instance type
instance_type = "t2.medium"

# Set the key pair name for SSH access
key_name = "cloudlabkey"

# docker build -t https://hub.docker.com/repository/docker/armandcismaru/deeptrader/deeptrader-tbse:latest .
# Set the security group ID to allow inbound traffic
security_group_id = "sg-0bb6339735da07a9a"

try:
    # s3 = boto3.resource("s3")
    # s3.create_bucket(Bucket="my-kops-bucket-fz19792")
    # s3.create_bucket(Bucket="input-data-fz19792")
    # s3.create_bucket(Bucket="output-data-fz19792")

    # s3_client = boto3.client("s3")
    # try:
    #     response = s3_client.upload_file(
    #         "markets.csv", "input-data-fz19792", "markets.csv"
    #     )
    # except ClientError as e:
    #     logging.error(e)

    # sys.exit()

    # Set the user data script to install Docker and Kubernetes
    user_data = """#!/bin/bash
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
    sudo apt-get update
    sudo apt-get install -y kubelet kubeadm kubectl
    sudo apt-get update
    sudo apt-get install python3 -y
    python -m pip install --upgrade pip
    pip install pylint
    pip install boto3
    pip install numpy
    pip install keras
    pip install matplotlib
    pip install progress
    pip install pandas
    """

    # kubectl create secret generic aws-credentials \
    # --from-literal=aws_access_key_id=<YOUR_ACCESS_KEY_ID> \
    # --from-literal=aws_secret_access_key=<YOUR_SECRET_ACCESS_KEY>

    # docker build -t armandcismaru/deeptrader:deeptrader-tbse .
    # docker push/pull/run armandcismaru/deeptrader:deeptrader-tbse

    # kubectl apply -f market-simulations.yaml

    node_count = 1
    # Create the EC2 instances
    instances = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=node_count,
        MaxCount=node_count,
        KeyName=key_name,
        SecurityGroupIds=[security_group_id],
        UserData=user_data,
    )

    # Print the instance IDs for reference
    instance_ids = [instance["InstanceId"] for instance in instances["Instances"]]
    print(f"Created instances with IDs: {', '.join(instance_ids)}")

    # Wait for instances to be running and get their IP addresses
    instance_ips = []
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=instance_ids)
    for instance_id in instance_ids:
        response = ec2.describe_instances(InstanceIds=[instance_id])
        instance_ips.append(
            response["Reservations"][0]["Instances"][0]["PublicIpAddress"]
        )

    print(f"Instance IPs: {', '.join(instance_ips)}")

    sys.exit()
    # Set up kops environment variables
    os.environ["KOPS_STATE_STORE"] = "s3://my-kops-bucket-fz19792"
    os.environ["AWS_REGION"] = region
    os.environ["NAME"] = "deeptradercluster.k8s.local"
    # os.system("export NAME=cluster.testikod.in")

    # reference = str(datetime.now())
    # r53_client = boto3.client("route53")
    # response = r53_client.create_hosted_zone(
    #     Name="testikod.in",
    #     CallerReference=reference,
    # )

    # kubectl create secret generic aws-creds --from-literal=AWS_ACCESS_KEY_ID=ASIA3MJWW2UGV3PJGQEH --from-literal=AWS_SECRET_ACCESS_KEY=4C6f30F3X0Wm6K6YVZDa5XcDNft2WU+kmN+fEtZH --from-literal=AWS_SESSION_TOKEN=FwoGZXIvYXdzECEaDFwXIy8jznRQOHAKhiLGAQpMWW80e5peYFH/EU6Nu/iuozOSyWcFELyDL0+fYojj4cWK2bms4ranmBLfJ9iCHKrjA8bKTmYI85a+r+kuthkUuDRxnGhg6JJmIVYoK/rvtzUxlYY989/WGrDDRFEn70Ap1n42i0SySgBZ0IQMom2XMsDkrgsIBcsO/Df6nmxLGaKhhg2h2QyunfL/xVRB4rrxDv2HtAshdSTP9mMVooh5GBlXLZnAfo2TIh0XwVEEsjaibNY8opnnE6pKPEttkvW0wCPQUCjwj/egBjIthFkx+P4dNxaYwtgHQSNAdX3jDyFPB3vnSq23w3EOMm9/IRwTHnTXqabC8BWq

    # Create the Kubernetes cluster using kops
    os.system(
        f"kops create cluster \
        --name deeptradercluster.k8s.local \
        --cloud aws \
        --zones {region}a,{region}b,{region}c \
        --node-count {node_count} \
        --node-size {instance_type} \
        --master-size m5.large \
        --dns-zone deeptradercluster.k8s.local \
        --topology private \
        --networking calico \
        --out s3://output-data-fz19702 \
        --yes"
    )

    print("Kubernetes cluster created")

    # Wait for the Kubernetes cluster to be ready
    # os.system("kops validate cluster --name cluster.testikod.in --wait 10m")

    # Print the Kubernetes cluster URL for reference
    # print("Kubernetes cluster URL:", os.environ["KOPS_STATE_STORE"])

except Exception as e:
    print(e)
    response = ec2.terminate_instances(
        InstanceIds=instance_ids,
    )
    print(response)
