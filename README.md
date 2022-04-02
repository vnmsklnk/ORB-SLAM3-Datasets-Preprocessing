# ORB-SLAM3-Datasets-Preprocessing
* At first you need to preprocess the dataset. Preprocessing includes the transformation of depth images into RGB images coordinates and converting pictures to undistorted format. You can do this by executing:
```
python preproccesing.py DEPTH_IMAGES_FOLDER COLOR_IMAGES_FOLDER INI_CONFIG_FILE
```
**WARNING**: You should have requirements installed for running script.

*NOTE*: You should specify config.ini file before running script. There is example of this file in the repository.

Your preproccesed files will be in ```depth_preprocessed``` and ```color_preprocessed``` folders.

* Then you can generate your own associations file executing:
```
python associate.py PATH_TO_RGB_FOLDER PATH_TO_DEPTH_FOLDER > associations.txt
```

*NOTE*: You can specify max_difference and timestamp2sec parameters. By default they equals 1000 microseconds and 1e6 accordingly.

* Now you can run ORB-SLAM3:
```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt SETTINGS_YAML_FILE PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```
*NOTE*: The repository already has a yaml configuration file for `multi_azure_recorded_circle/1m` dataset.