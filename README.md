# W4995 deep learning project (Speaker Verification)

By: Yue Luo(yl4003), Bingqing Wei(bw2581), Gleb Vizitiv(gv2256)

This project is built based on the repository at 'https://github.com/Janghyun1230/Speaker_Verification'

Using Google Cloud for computation.
To log in the VM, start the vm instance `dl1`, and

` $ gcloud compute --project "w4995-dl-proj" ssh --zone "us-central1-c" "dl1" `

File Transferring:

`$ gcloud compute scp [LOCAL_FILE_PATH] [INSTANCE_NAME]:~`

`$ gcloud compute scp --recurse [INSTANCE_NAME]:[REMOTE_DIR] [LOCAL_DIR]`

`$ gutils cp [-r] gs://[BUCKET_NAME] [LOCAL_Name]`

`$ gsutil cp -r gs://sv-proj/voxceleb . `

### Prerequisites
Software version/hardware settings we use.

- 4 vCPUs, 16 GB RAM
- 300 GB SSD [check with df -h]
- 1 Nvidia Tesla K80(12GB memory)  [check with nvidia-smi]
- CUDA v10 [nvcc --version], CUDNN v7 [ls /usr/local/cuda/lib64/ | grep cudnn]

And
- Python 3.5.3
- Tensorflow 1.13.1
- numpy 1.16.2
- librosa 0.6.3


### Preparation
1. Get the code.
   ```Shell
   git clone https://github.com/lawy623/dl_proj.git
   cd dl_proj
   ```

 2. Download the raw dataset. We use [Voxceleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) for this project.
 Go into `./raw_data` and run `$sh get_data_voxceleb.sh`. We only use the training dataset and separate it for our testing. It is about 37GB large.

### Data Preprocess
Run `python src/data.py` for data preprocessing.

Some statistics: 1211 speakers. 0.8/0.1/0.1 -> [Train(969)/ Valid(121)/ Test(121)]. Min(nb_utter)=45. Max(nb_utter)=1002. Not all the data will be use for testing and validation,
only a partial fixed set will be used.

### Training
Run `python src/main.py` for training. If you want to specify the location that stores the check point, doing it by `python src/main.py --model_path [MODEL_PATH]`.


### Testing
Run `python src/main.py --mode 'test'` for testing. If you want to specify the location that stores the check point, as well as the checkpoint index,
doing it by `python src/main.py --mode 'test' --model_path [MODEL_PATH] --iter [idx]`.
