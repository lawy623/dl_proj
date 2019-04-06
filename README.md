# dl_proj
w4995 deep learning project (Speaker Verification)


Using Google Cloud for computation.
To log in the VM, start the vm instance `dl1`, and
` $ gcloud compute --project "w4111db" ssh --zone "us-east1-b" "cs4111" `

software version/hardware settings.

- 4 vCPUs, 16 GB RAM
- 100 GB SSD [check with df -h]
- 1 Nvidia Tesla P100(16GB memory)  [check with nvidia-smi]
- CUDA v10 [nvcc --version], CUDNN v7 [ls /usr/local/cuda/lib64/ | grep cudnn]
- IP: 35.243.154.118

- Python
- Tensorflow
-
