# ORB-SLAM3-Datasets-Preprocessing
* At first you need to preprocess the dataset. Preprocessing includes the transformation of depth images into RGB images coordinates and converting pictures to undistorted format. You can do this by executing:
```
python preprocessing.py PATH_TO_DATASET PATH_TO_SAVE_COLOR PATH_TO_SAVE_DEPTH INI_CONFIG_FILE
```
**WARNING**: You should have requirements installed for running script.

*NOTE*: You should specify `preprocessing_config.ini` file before running script. There is example of this file in the repository.

*NOTE*: You should have `color` and `depth` folders in dataset.

* Then you can generate your own associations file for one-view or two-view cases:
```
python associate_one_view.py RGB_DIR DEPTH_DIR > associations.txt
```
```
python associate_two_view.py RGB_MASTER_DIR DEPTH_MASTER_DIR RGB_SLAVE_DIR DEPTH_SLAVE_DIR > associations.txt
```

*NOTE*: You can specify max_difference and timestamp2sec parameters. By default they equals 1000 microseconds and 1e6 accordingly.

* Now you can run ORB-SLAM3.
For one-view version use:
```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt SETTINGS_YAML_FILE PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```
For two-view:
```
./Examples/RGB-D-Two-View/rgbd_tum_tw Vocabulary/ORBvoc.txt SETTINGS_YAML_FILE PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```
*NOTE*: The repository already has a yaml configuration file example. All `Slave` parameters and `TransformationMatrix` don't affect anything in the case of one-view version.
