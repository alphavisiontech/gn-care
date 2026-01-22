# Automatic Reset Camera 
*Updated: January 22nd, 2026*

### Instructions 
To enable the automatic reset camera function, follow these instructions:

1. Ensure that the `cam_utils.py` is in the `utils` directory 
    * If it doesn't exist, download it from the GitHub Repo [./utils/cam_utils.py]
2. Open `main_rknn.py`, ensure that `open_cam()` and `reset_usb_devices()` functions are called appropriately 
    * If they are not called appropriately, please modify the cam 
3. Open the terminal
4. Type `sudo visudo`
5. Write these lines of code at the BOTTOM of the page (order matters in `visudo`):
    orangepi ALL=(ALL) NOPASSWD: /usr/bin/modproble, /sbin/modprobe
    orangepi ALL=(ALL) NOPASSWD: /usr/bin/tee, /bin/tee
    orangepi ALL=(ALL) NOPASSWD: /usr/bin/udevadm, /sbin/udevadm
    orangepi ALL=(ALL) NOPASSWD: /usr/bin/pkill
    orangepi ALL=(ALL) NOPASSWD: /usr/bin/bash, /bin/bash
    orangepi ALL=(ALL) NOPASSWD: /usr/bin/echo, /bin/echo
6. Save the file by pressing `ctrl + x`, then `y`, then `ctrl + m`
7. The camera should reset automatically every time it encounters a problem 

### Troubleshooting 
1. The terminal still prompts password when restarting the camera 
    * Type `sudo visudo` in the terminal 
    * Check if the lines of code are the same with step 5. 
    * Ensure that the username is correct. 
        - You can find the username in the terminal. The first word of the terminal before the `@` tag is your username 
        - By default, the username of an OrangePi device is `orangepi`. If the username you found is NOT `orangepi`, then change the `orangepi` of step 5 into the username you have found 
    * Ensure that the lines on step 5 are written at the BOTTOM of the page 
