wget --user voxceleb1902 --password nx0bl2v2 http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa
wget --user voxceleb1902 --password nx0bl2v2 http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab
wget --user voxceleb1902 --password nx0bl2v2 http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac
wget --user voxceleb1902 --password nx0bl2v2 http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad
#wget --user voxceleb1902 --password nx0bl2v2 http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
cat vox1_dev* > vox1_dev_wav.zip
rm vox1_dev_wav_parta*

mkdir voxceleb
unzip vox1_dev_wav.zip -d voxceleb/
rm vox1_dev_wav.zip
