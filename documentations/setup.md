# Setup New Device Manual 
*Updated: January 21st, 2026* 

1. If your OrangePi has been rerooted from scratch or newly bought, ensure that there’s nothing inside of the OrangePi and the system is running with default setting 
2. Pull the **gn-care** repository into your local OrangePi 
3. Download Anaconda from the official website and install it on OrangePi 
    - Download Anaconda distribution (and NOT the miniconda3)
    - Choose the Linux with ARM architecture package and download it
    - Once downloaded, locate the directory where the Anaconda package is downloaded 
    - Open the terminal and go to the directory you found on step C and type (and <u>*replace the distribution file name*</u> with the correct one) 
        `bash Anaconda-distribution-name.pkg`
    - Follow the official instruction to install the Anaconda
    - Once completed, you will be asked to activate the init shell. Please press ‘NO’ and complete the installation 
4. Activate the conda session by typing
    `source /home/orangepi/anaconda3/bin/activate` 
    - If it’s activated properly, there should be a `(base)` at the very front of the line in the terminal 
5. Create a virtual environment named fd using `conda` 
    - Open the terminal and type (and <u>*replace the path*</u> with the correct one) 
        `conda create -n fd python=3.9`
    - Once successfully created, activate the environment by typing 
        `conda activate fd`
    - Ensure that your terminal is currently in the virtual environment by checking the very front of the line. It should have `(fd)`
    - Install other dependencies: 
        `pip install opencv-python=4.10.0.84 stomp-py==8.2.0`
6. Install the RKNN-Toolkit by following the instructions below. 
    - Locate the directory of **env_files**
    - In the terminal (needs to be in the `fd` virtual environment), type the following (and <u>*replace the paths and file name*</u> with the correct one) 
        `cd /path/to/rknn-toolkit_package.whl`
        `pip install rknn_toolkit_package_name.whl `
    - You should have successfully installed RKNN-Toolkit
7. Remove the in-built **librknnrt.so** (NOT the one inside **env_files**) in the **/usr/lib** folder by following the instructions below 
    - Ensure that you are in the **root** directory 
    - Remove the **librknnrt.so** file by typing the following command on the terminal 
        `sudo rm /usr/lib/librknnrt.so`
    - Enter the password of orangepi and press “Enter” 
    - Check that the file is properly removed by typing the following command on the terminal 
        `cd /usr/lib`
        `ls`
    - You should NOT see any **librknnrt.so** file 
8. Locate the directory of **librknnrt.so** file in the **env_files** directory and move the **librknnrt.so** to the **/usr/lib** folder by typing the following command on the terminal (and <u>*replace the paths and file name*</u> with the correct one) 
    `sudo mv /path/to/your/librknnrt.so /usr/lib/librknnrt.so`
9. Change the permissions of the **librknnrt.so** by typing the following command on the terminal
    `chmod +x librknnrt.so`
    - NOTE: you should be in the **/usr/lib** directory for the command to work 
10. Ensure that everything is correct by following the instructions below
    - In the terminal, ensure that you are at **/usr/lib** directory and type the command
        `ls`
    - You should see the **librknnrt.so** existing and that the color of the file is green in bold font.
11. Open the **startup.sh** file and ensure that the first two lines are 
    `source /home/orangepi/anaconda3/bin/activate `
    `conda activate fd`
12. Ensure that the fall detection script can run properly
    - Ensure that you are still on the `fd` environment 
    - Ensure that you are on the **gn-care** directory 
    - Type the following command
        `python main_rknn.py --input-type webcam --score-threshold 0.6 --keypoint-threshold 0.3 --fall-consecutive-threshold 10 --ratio-threshold 1.2 --backward-ratio-threshold 0.6 --show-display`
    - If a screen shows the camera result appears, ensure that the keypoints are properly shown on top of your body
    - Quit the program by typing `ctrl + c`
    - The script is already configured properly and you can quit the script. 
13. Follow the [Auto-restart](startup.md) instruction to configure the OrangePi to run the falling detection script on reboot. 
14. To enable the automatic restart of the camera in case of sudden disconnection of the camera to the OrangePi, please follow the [Restarting Camera](reset_cam.md) instruction. 
15. Restart the OrangePi and everything should run normally. 
