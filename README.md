# ORB-SLAM3-Datasets-Preprocessing
* At first you need to preprocess the dataset. You can do this by executing:
```
python preproccesing.py PATH_TO_DEPTH_IMAGES_FOLDER PATH_TO_COLOR__IMAGES FOLDER CONFIG_FILE_NAME.ini
```
Your preproccesed file will be in ```depth_preprocessed``` and ```color_preprocessed``` folders.

* Then you can generate your own associations file executing:
```
python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```
**WARNING**: You must have ```color``` and ```depth``` folder with images next to timestamps text files.

*NOTE*: You can specify max_difference parameter. By default it equals 1000 microseconds.

* Now you can run ORB-SLAM3:
```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt PATH_TO_SETTINGS_YAML_FILE PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```