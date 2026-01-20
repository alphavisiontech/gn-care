import os 
import cv2
import sys 
import subprocess
import time
import logging
import glob
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logging',
    filemode='a',
)

def find_usb_camera_devices():
    """Find all available USB camera devices and their corresponding USB paths"""
    video_devices = []
    usb_paths = []
    
    # Find all video devices
    for video_dev in glob.glob('/dev/video*'):
        try:
            # Get device info using v4l2-ctl if available
            result = subprocess.run(f"v4l2-ctl --device={video_dev} --info", 
                                  shell=True, capture_output=True, text=True)
            if "usb" in result.stdout.lower() or result.returncode == 0:
                video_devices.append(video_dev)
        except:
            # Fallback: assume it's a camera if the device exists
            video_devices.append(video_dev)
    
    # Find corresponding USB paths
    for video_dev in video_devices:
        try:
            # Get the real path and extract USB info
            real_path = os.path.realpath(video_dev)
            # Look for USB device path in /sys/class/video4linux/
            video_name = os.path.basename(video_dev)
            sys_path = f"/sys/class/video4linux/{video_name}/device"
            
            if os.path.exists(sys_path):
                usb_path = os.path.realpath(sys_path)
                # Extract USB bus-port format (e.g., "7-1")
                usb_match = re.search(r'/usb\d+/(\d+-[\d.]+)', usb_path)
                if usb_match:
                    usb_paths.append(usb_match.group(1))
                else:
                    usb_paths.append(None)
            else:
                usb_paths.append(None)
        except:
            usb_paths.append(None)
    
    return list(zip(video_devices, usb_paths))

def wait_for_device_enumeration(timeout=30):
    """Wait for USB devices to be enumerated after module reload"""
    print("Waiting for device enumeration...")
    logging.info("Waiting for device enumeration...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if any video devices appeared
        if glob.glob('/dev/video*'):
            time.sleep(2)  # Give it a bit more time to stabilize
            return True
        time.sleep(1)
    
    print("Timeout waiting for device enumeration")
    logging.warning("Timeout waiting for device enumeration")
    return False

def reset_usb_controller():
    """Reset the entire USB controller/hub - more aggressive approach"""
    print("Resetting USB controller...")
    logging.warning("Resetting USB controller...")
    
    try:
        # Reset all USB devices by rescanning the USB bus
        subprocess.run("sudo bash -c 'echo 1 > /sys/bus/usb/devices/usb*/authorized'", shell=True)
        time.sleep(1)
        subprocess.run("sudo bash -c 'echo 0 > /sys/bus/usb/devices/usb*/authorized'", shell=True)
        time.sleep(2)
        subprocess.run("sudo bash -c 'echo 1 > /sys/bus/usb/devices/usb*/authorized'", shell=True)
        time.sleep(3)
        
        # Alternative: Reset USB ports
        subprocess.run("sudo udevadm trigger --subsystem-match=usb", shell=True)
        subprocess.run("sudo udevadm settle", shell=True)
        time.sleep(2)
        
        print("USB controller reset complete")
        logging.info("USB controller reset complete")
        
    except Exception as e:
        print(f"USB controller reset failed: {e}")
        logging.error(f"USB controller reset failed: {e}")

def reset_usb_devices(usb_device_path): 
    if not usb_device_path:
        print("No USB device path provided, skipping USB reset")
        logging.warning("No USB device path provided, skipping USB reset")
        return
        
    # resetting the device by unbinding and binding the usb device
    print(f"Reset USB device {usb_device_path}...")
    logging.warning(f"Reset USB device {usb_device_path}...")
    subprocess.run(f"echo '{usb_device_path}' | sudo tee /sys/bus/usb/drivers/usb/unbind", shell=True)
    time.sleep(2)
    subprocess.run(f"echo '{usb_device_path}' | sudo tee /sys/bus/usb/drivers/usb/bind", shell=True)
    time.sleep(3)  # Increased wait time
    print("Reset complete")
    logging.info("Reset Complete")

def reload_cam_module():
    """Reloading the uvcvideo module with enhanced recovery"""
    
    print("Reloading the video kernel module...")
    logging.warning("Reloading the video kernel module...")
    
    # Kill any processes that might be using the camera
    subprocess.run("sudo pkill -f 'v4l\\|video'", shell=True)
    time.sleep(1)
    
    # remove module (but can fail if something is using the camera, or if cap is not released properly)
    subprocess.run("sudo modprobe -r uvcvideo", shell=True)
    time.sleep(3)
    
    # Also try to remove related modules for a cleaner restart
    subprocess.run("sudo modprobe -r videobuf2_v4l2 videobuf2_common", shell=True)
    time.sleep(2)
    
    # loading the module again
    subprocess.run("sudo modprobe uvcvideo", shell=True)
    time.sleep(3)
    
    # rescan devices with more comprehensive approach
    subprocess.run("sudo udevadm trigger --subsystem-match=video4linux", shell=True)
    subprocess.run("sudo udevadm trigger --subsystem-match=usb", shell=True)
    subprocess.run("sudo udevadm control --reload", shell=True)
    subprocess.run("sudo udevadm settle", shell=True)
    time.sleep(3)
    
    # Wait for device enumeration
    if not wait_for_device_enumeration():
        # If normal enumeration fails, try USB controller reset
        reset_usb_controller()
        wait_for_device_enumeration()
    
    print("module reloaded and rescan completed")
    logging.info("module reloaded and rescan completed")

def find_working_camera():
    """Find the first working camera device"""
    camera_devices = find_usb_camera_devices()
    
    # Add debug info
    print(f"Found {len(camera_devices)} potential camera devices")
    logging.info(f"Found {len(camera_devices)} potential camera devices")
    
    for video_dev, usb_path in camera_devices:
        print(f"Testing camera device: {video_dev} (USB: {usb_path})")
        if os.path.exists(video_dev):
            # Try to open the camera briefly to test if it works
            test_cap = cv2.VideoCapture(video_dev, cv2.CAP_V4L2)
            if test_cap.isOpened():
                test_cap.release()
                print(f"Found working camera: {video_dev} (USB: {usb_path})")
                logging.info(f"Found working camera: {video_dev} (USB: {usb_path})")
                return video_dev, usb_path
            test_cap.release()
        else:
            print(f"Device {video_dev} does not exist")
    
    print("No working camera found")
    logging.error("No working camera found")
    return None, None

def check_usb_devices():
    """Check what USB devices are actually connected"""
    try:
        result = subprocess.run("lsusb", shell=True, capture_output=True, text=True)
        print("Connected USB devices:")
        print(result.stdout)
        logging.info(f"USB devices: {result.stdout}")
        
        # Also check for video devices specifically
        video_devices = glob.glob('/dev/video*')
        print(f"Available video devices: {video_devices}")
        logging.info(f"Available video devices: {video_devices}")
        
    except Exception as e:
        print(f"Failed to check USB devices: {e}")
        logging.error(f"Failed to check USB devices: {e}")

def open_cam():
    retry_count = 0
    module_reload_count = 0
    
    while True:
        # Add debug information
        check_usb_devices()
        
        # Dynamically find available camera
        video_dev, usb_path = find_working_camera()
        
        if not video_dev:
            print("No camera device found")
            logging.error("No camera device found")
            
            if retry_count >= 5 and module_reload_count < 3:
                print(f"Reloading camera module (attempt {module_reload_count + 1}/3)")
                reload_cam_module()
                module_reload_count += 1
                retry_count = 0
            elif module_reload_count >= 3:
                print("Maximum module reload attempts reached. Trying USB controller reset...")
                reset_usb_controller()
                module_reload_count = 0
                retry_count = 0
            else:
                retry_count += 1
                
            time.sleep(5)  # Increased wait time
            continue
            
        cap = cv2.VideoCapture(video_dev, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Camera opened: {video_dev}")
            logging.info(f"Camera opened: {video_dev}")
            return cap, video_dev, usb_path
        else: 
            print(f"Failed to open camera {video_dev}. Resetting the USB and retrying...")
            logging.error(f"Failed to open camera {video_dev}. Resetting the USB and retrying...")
            cap.release()
            reset_usb_devices(usb_path)
            retry_count += 1
            if retry_count >= 10: 
                reload_cam_module()
                retry_count = 0
                module_reload_count += 1
            time.sleep(5)

def cam_stream():
    cap, current_video_dev, current_usb_path = open_cam()
    
    try: 
        while True: 
            ret, frame = cap.read()
            if not ret or frame is None: 
                print("No frame. Resetting USB and reconnecting...")
                logging.warning("No frame. Resetting USB and reconnecting...")
                cap.release()
                reset_usb_devices(current_usb_path)
                cap, current_video_dev, current_usb_path = open_cam()
                continue
                
            # cv2.imshow("Cam Stream", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    finally: 
        if cap is not None: 
            cap.release()
        cv2.destroyAllWindows()
        print("Resources all released")
        logging.info("Resources all released")
        return cap

