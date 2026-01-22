# Auto-Restart Manual 
*Updated: January 21st, 2026*

### Instructions 

1. Ensure that the **gn-care** folder is inside the **Desktop** 
2. Open the terminal 
3. Type `sudo nano /etc/systemd/system/fall_detection.service`
4. Write the following script:
    [Unit]
    Description=Fall Detection Script
    After=network.target

    [Service]
    Type=simple
    User=orangepi
    WorkingDirectory=/home/orangepi/Desktop/gn-care
    ExecStart=/home/orangepi/Desktop/gn-care/startup.sh
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
5. Press `ctrl + x`, then `y`, then `ctrl + m`. 
6. In the terminal, type `sudo systemctl daemon-reload`
7. Enter the password for orangepi.
8. Then, type `sudo systemctl enable fall_detection`
0. Lastly, type `sudo systemctl start fall_detection`

### Verify
To verify whether the auto-restart function is running properly, follow the following instructions:

1. Reboot the orangepi hub 
2. Open the terminal 
3. Type `ps aux | grep sleep` 
    * A `sleep 60s` should appear in the listed processes (NOTE: the `60s` should come from the *startup.sh* file. In the case that the time is different, verify whether the *startup.sh* file has `60s` or a different timing.)
    * If it doesn't appear, then there's a mistake or error 
4. Wait for 1 minute
5. Type `ps aux | grep python`
    * A `python fall_detection_latest.py` should appear in the listed processes
    * If it doesn't appear, then there's a mistake or error. Refer to the [Troubleshooting](#troubleshooting) section. 

### Troubleshooting 

1. `sleep 60s` appears but keeps repeating at different time and `python main_rknn.py` doesn't appear 
    * Steps:
        - Open *startup.sh* file
        - Ensure that the first 3 lines are: 
            `
            #!/bin/bash
            source /home/orangepi/anaconda3/bin/activate
            conda activate fd
            `
        - Ensure that there's NO `--show-display` tag on line 6
    * If there are changes, then save the changes and reboot the orangepi 
    * Ff it still doesn't work even after saving the changes and rebooting the orangepi, comment out line 503. Then, reboot it once more. 
    
2. `sleep 60s` doesn't appear at all
    * In the terminal, type `sudo systemctl status fall_detection`. This will show you the status of the process
        - Solution 1: 
            - See if the process is active or inactive. 
            - If it is active, then type `sudo nano /etc/systemd/system/fall_detection.service`
                > On the bottom of the page, see if there's an error line saying "Bad lock file is ignored:<file-name>" where <file-name> is the name of the file written
                > Copy the <file-name> 
                > Press `ctrl + x`
                > Then, type `sudo rm <file-name>`. Replace <file-name> with the actual file name that you have copied 
                > Then, type `sudo nano /etc/systemd/system/fall_detection.service` again
                > Ensure that the script from step 12 still exists, and the error line is gone
                > Press `ctrl + x` 
            - Type `sudo systemctl restart fall_detection` 
            - Type `ps aux | grep sleep`
            - The `sleep 300s` should appear 
        
        - Solution 2: 
            - See if there's any error code displayed, such as the text in red color or a error 203 code
            - If this is the case, then there are several reasons that this happen:
                > Something wrong with the *startup.sh* file. Ensure that it is written correctly
                > Bad lock file just like solution 1. Please refer to solution 1
                > Something wrong with `main_rknn.py`
            
        - Solution 3: 
            - In the terminal, type `journalctl -u fall_detection`
            - See what's wrong and troubleshoot it from there

        

