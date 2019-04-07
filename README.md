# W4995 deep learning project (Speaker Verification)

By: Yue Luo(yl4003), Bingqing Wei(bw2581), Gleb Vizitiv(gv2256)


Using Google Cloud for computation.
To log in the VM, start the vm instance `dl1`, and

` $ gcloud compute --project "w4995-dl-proj" ssh --zone "us-east1-b" "dl1" `

File Transferring:
`$ gcloud compute scp [LOCAL_FILE_PATH] [INSTANCE_NAME]:~`

`$ gcloud compute scp --recurse [INSTANCE_NAME]:[REMOTE_DIR] [LOCAL_DIR]`

`$ gutils cp [-r] gs://[BUCKET_NAME] [LOCAL_Name]`

There is a preprocessed Voxceleb dataset in the BUCKET.
`$ gsutil cp -r gs://sv-proj/voxceleb . `

### Prerequisites
software version/hardware settings we use.

- 4 vCPUs, 16 GB RAM
- 100 GB SSD [check with df -h]
- 1 Nvidia Tesla P100(16GB memory)  [check with nvidia-smi]
- CUDA v10 [nvcc --version], CUDNN v7 [ls /usr/local/cuda/lib64/ | grep cudnn]
- IP: 35.243.154.118

And
- Python 3.5.3
- Tensorflow 1.13.1
- numpy 1.16.2


### Installation
1. Get the code.
   ```Shell
   git clone https://github.com/lawy623/dl_proj.git
   cd dl_proj
   ```

 2. Download the raw dataset. We use [Voxceleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) for this project.
 Go into `./raw_data` and run `$sh get_data_voxceleb.sh`. We only use the training dataset and separate it for our testing. It is about 37GB large.
