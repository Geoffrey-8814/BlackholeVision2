# BlackholeVision2
A FRC vision system

# **BlackholeVision Setup Guide**  

## **Step 1: Setup Your Device**  
1. Install **Ubuntu** on your device.  

## **Step 2: Install Conda**  
1. Download and install **Conda** or **Anaconda**:  
   👉 [Download here](https://www.anaconda.com/download)  
2. Create a **Python 3.11** environment:  
   ```bash
   conda create -n blackhole_vision python=3.11
   ```  

## **Step 3: Clone the Repository**  
1. Open a terminal and run:  
   ```bash
   git clone https://github.com/Geoffrey-8814/BlackholeVision2.git
   ```  

## **Step 4: Install Required Libraries**  

1. **Install nano (text editor):**  
   ```bash
   sudo apt update
   sudo apt install nano
   ```  
2. **Activate your Conda environment:**  
   ```bash
   conda activate blackhole_vision
   ```  
3. **Navigate to the project directory:**  
   ```bash
   cd BlackholeVision2
   ```  
4. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  
5. **Install RobotPy (for FRC support):**  
   ```bash
   python3 -m pip install --extra-index-url=https://wpilib.jfrog.io/artifactory/api/pypi/wpilib-python-release-2025/simple robotpy
   ```  
6. **Install PyTorch:**  
   - **With CUDA 11.8:**  
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```  
   - **With CUDA 12.6:**  
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
     ```  
   - **Without CUDA:**  
     ```bash
     pip3 install torch torchvision torchaudio
     ```  

---

## **Step 5: Setup USB Ports**  

### **1. Identify Camera USB Ports**  
1. Connect a camera to your device.  
2. Run the following command to get the USB port details:  
   ```bash
   udevadm info --name=/dev/video0 --attribute-walk
   ```  
3. Look for an **ID** that looks like `/usb1/1-2/1-2.1/1-2.1:1.0`.  
4. Extract the **port ID** (e.g., `1-2.1` from `1-2.1:1.0`).  

### **2. Create and Edit USB Rules**  
1. Open the USB rules file:  
   ```bash
   sudo nano /etc/udev/rules.d/99-usb-cameras.rules
   ```  
2. Add the following rule, replacing **1-2.1** with your port ID and **PORTNAME** with your desired name:  
   ```bash
   SUBSYSTEM=="video4linux", KERNELS=="1-2.1", SYMLINK+="PORTNAME"
   ```  
3. Save and exit (`CTRL+X`, then `Y`, then `Enter`).  

### **3. Apply the New Rules**  
```bash
sudo udevadm control --reload-rules
```  

### **4. Verify the Assigned Port**  
```bash
ls -l /dev/PORTNAME
```  
- If a camera is detected, the port assignment is successful.  
- Repeat these steps for additional cameras.  

---

## **Step 6: Run `sudo` Without Password**  

1. Open a terminal and edit the **sudoers** file:  
   ```bash
   sudo visudo
   ```  
2. Scroll to the bottom and add this line, replacing **$USER** with your actual username:  
   ```bash
   $USER ALL=(ALL) NOPASSWD: ALL
   ```  
3. Save and exit (`CTRL+X`, then `Y`, then `Enter`).  
4. Open a new terminal and test:  
   ```bash
   sudo visudo
   ```  
   - If **no password is required**, the setup is successful.  

---

## **Step 7: Create a Setup Script**  

1. Open a terminal and create a script:  
   ```bash
   nano ~/startBlackholeVision.sh
   ```  
2. Add the following lines:  
   ```bash
   # Navigate to the project directory
   cd BlackholeVision2
   
   # Apply USB rules
   sudo udevadm trigger

   # Run the program (replace with your Conda Python path)
   /path/to/conda/envs/blackhole_vision/bin/python __init__.py
   ```  
3. Save and exit (`CTRL+X`, then `Y`, then `Enter`).  

4. Make the script executable:  
   ```bash
   sudo chmod +x ~/startBlackholeVision.sh
   ```  

---

## **Step 8: Add BlackholeVision to Startup Applications**  

1. Open **Startup Applications** on Ubuntu.  
2. Click **Add**.  
3. Enter the command:  
   ```bash
   ./startBlackholeVision.sh
   ```  
4. Click **Add** to confirm.  

---

## **Step 9: Enable Automatic Login**  

Follow [this guide](https://help.ubuntu.com/stable/ubuntu-help/user-autologin.html.en) or:  

1. Open **Settings** > **System**.  
2. Select **Users**.  
3. Click **Unlock** (top-right corner) and enter your password.  
4. Select the user account you want to log in automatically.  
5. Enable **Automatic Login**.  

---

🚀 **Setup Complete!** Now your **BlackholeVision** system will start automatically upon boot. 🚀