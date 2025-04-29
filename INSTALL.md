
# Install
1. Clone the project

    ```Shell
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2
    cd Ultra-Fast-Lane-Detection-V2
    ```

2. Create a conda virtual environment and activate it

    ```Shell
    conda create -n lane-det python=3.7 -y
    conda activate lane-det
    ```

3. Install dependencies

    ```Shell
    # If you dont have pytorch
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

    pip install -r requirements.txt

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
    # Install Nvidia DALI (Very fast data loading lib))

    #Next, install the needed CUDA-Toolkit library from this url https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local.
    #This will give you the correct version of CUDA-Toolkit to run the program

    #This solution to fix g++ is only avaliable for ubuntu/wsl and might not work in other distros
    sudo nano /etc/apt/sources.list

    #Enter the following line inside the file
    deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe

    #Make sure to exit and save
    apt update
    apt install g++-7

    export CXX=/usr/bin/g++-7
    export CC=/usr/bin/gcc-7

    cd my_interp

    sh build.sh

    #This next line is to correctly tell the program where to find libcuda.so to avoid a runtime error
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

    #This next line is only useful for WSL to disalbe NVML to avoid a runtime error
    export DALI_DISABLE_NVML=1 
    ```

4. Data preparation
    #### **4.1 Tusimple dataset**
    Download [CULane](https://xingangpan.github.io/projects/CULane.html), [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3), or [CurveLanes](https://github.com/SoulmateB/CurveLanes) as you want. The directory arrangement of Tusimple should look like(`test_label.json` can be downloaded from [here](https://github.com/TuSimple/tusimple-benchmark/issues/3) ):
    ```
    $TUSIMPLE
    |──clips
    |──label_data_0313.json
    |──label_data_0531.json
    |──label_data_0601.json
    |──test_tasks_0627.json
    |──test_label.json
    |──readme.md
    ```
    For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

    ```Shell
    python scripts/convert_tusimple.py --root /path/to/your/tusimple

    # this will generate segmentations and two list files: train_gt.txt and test.txt
    ```
    #### **4.2 CULane dataset**
    The directory arrangement of CULane should look like:
    ```
    $CULANE
    |──driver_100_30frame
    |──driver_161_90frame
    |──driver_182_30frame
    |──driver_193_90frame
    |──driver_23_30frame
    |──driver_37_30frame
    |──laneseg_label_w16
    |──list
    ```
    For CULane, please run:
    ```Shell
    python scripts/cache_culane_ponits.py --root /path/to/your/culane

    # this will generate a culane_anno_cache.json file containing all the lane annotations, which can be used for speed up training without reading lane segmentation maps

    # note that if this fails due to the program not having writting access to the program, then you can use the following command to fix it. There is likey a better way of allowing writing privildges, but this worked for when we were working with it

    sudo chmod 777 ~/path/to/your/CULane 

    ```
    #### **4.3 CurveLanes dataset**
    The directory arrangement of CurveLanes should look like:
    ```
    $CurveLanes
    |──test
    |──train
    |──valid
    ```
    For CurveLanes, please run:
    ```Shell
    python scripts/convert_curvelanes.py --root /path/to/your/curvelanes

    python scripts/make_curvelane_as_culane_test.py --root /path/to/your/curvelanes

    # this will also generate a curvelanes_anno_cache_train.json file. Moreover, many .lines.txt file will be generated on the val set to enable CULane style evaluation.
    ```

5. Install CULane evaluation tools (Only required for testing). 

    If you just want to train a model or make a demo, this tool is not necessary and you can skip this step. If you want to get the evaluation results on CULane, you should install this tool.

    This tools requires OpenCV C++. Please follow [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to install OpenCV C++. ***When you build OpenCV, remove the paths of anaconda from PATH or it will be failed.***
    ```Shell
    # First you need to install OpenCV C++. 
    # After installation, make a soft link of OpenCV include path.

    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
    ```
    We provide three kinds of complie pipelines to build the evaluation tool of CULane.

    Option 1:

    ```Shell
    cd evaluation/culane
    make
    ```

    Option 2:
    ```Shell
    cd evaluation/culane
    mkdir build && cd build
    cmake ..
    make
    mv culane_evaluator ../evaluate
    ```

    For Windows user:
    ```Shell
    mkdir build-vs2017
    cd build-vs2017
    cmake .. -G "Visual Studio 15 2017 Win64"
    cmake --build . --config Release  
    # or, open the "xxx.sln" file by Visual Studio and click build button
    move culane_evaluator ../evaluate
    ```

6. Modifyig data_root

    For whatever reason, whenever test.py, train.py, and other python files were run, the cfg would configure the data_root incorrectly leading to the program not being able to find the dataset. To fix this, one must edit the files directly and add in the path to where the dataset files are.

    ```Shell
    #The following lines will look as follows in train.py (line 52), test.py (line 9), and analyze_weather_clusters.py (line 50)
    cfg.data_root = ''

    #To correctly fix this replace the empty space to the location of the dataset you are using. Use the following as an example of how one of our memebers did it
    cfg.data_root = '/home/andrew/CULane/CULane/'
    ```