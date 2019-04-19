pip install -r requirements.txt

cd Mask_RCNN
pip install -r requirements.txt
python3 setup.py install

cd ..
git submodule --init --update cocoapi

cd cocoapi/PythonAPI
make
python setup.py install

pwd
cd ../../cycleGan-pix2pix
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
pip install -r requirements.txt

cd ..
