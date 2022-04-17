# ORB-SLAM3-Datasets-Preprocessing
* At first you need to preprocess the dataset. Preprocessing includes the transformation of depth images into RGB images coordinates and converting pictures to undistorted format. You can do this by executing:
```
python preprocessing.py PATH_TO_DATASET PATH_TO_SAVE_COLOR PATH_TO_SAVE_DEPTH INI_CONFIG_FILE
```
**WARNING**: You should have requirements installed for running script.

*NOTE*: You should specify config.ini file before running script. There are examples of this file for master and slave cameras in the repository.

*NOTE*: You should have `color` and `color` folders in dataset.

* Then you can generate your own associations file executing:
```
python associate.py PATH_TO_RGB_MASTER PATH_TO_DEPTH_MASTER PATH_TO_RGB_SLAVE PATH_TO_DEPTH_SLAVE > associations.txt
```

*NOTE*: You can specify max_difference and timestamp2sec parameters. By default they equals 1000 microseconds and 1e6 accordingly.

* Now you can run ORB-SLAM3:
```
./Examples/RGB-D-Two-View/rgbd_tum_tw Vocabulary/ORBvoc.txt SETTINGS_YAML_FILE PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```
*NOTE*: The repository already has a yaml configuration file for `multi_azure_recorded_circle` dataset.