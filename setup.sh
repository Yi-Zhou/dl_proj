git submodule update --init --recursive cocoapi
cd cocoapi/PythonAPI
make
cd ../../cycleGan-pix2pix
#bash datasets/download_cyclegan_dataset.sh horse2zebra
cd ../Mask_RCNN
/opt/anaconda3/bin/pip install -r requirements.txt
python3 setup.py install
