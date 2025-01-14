# Installing Python on Windows or Linux by desired version

## Selecting and Installing Python Version

To install Python on your system (Windows or Linux), follow the steps below to select and install your desired version.

Visit the official Python download page to find all available versions: [https://www.python.org/downloads/](https://www.python.org/downloads/). This page lists all current and past versions of Python, allowing you to select a specific version that suits your requirements.

## Windows Installation

1. **Download Python:**

   - Go to the [Python download page](https://www.python.org/downloads/).
   - Browse through the available versions and select the one you wish to install (e.g., Python 3.9, Python 3.10).
   - Click on the **Windows Installer** corresponding to the version you selected (you can choose between 32-bit and 64-bit).

2. **Install Python:**

   - Run the downloaded `.exe` file.
   - Make sure to check the box **"Add Python to PATH"** during the installation process to make Python accessible from the command line.
   - Click on **Customize installation** if you need to choose specific features or change the installation path.

3. **Verify Installation:**
   - Open **Command Prompt** and run:
     ```sh
     python --version
     ```
   - This should display the installed Python version.

## Linux Installation

1. **Update Package List:**

   - Run the following command to update the package list:
     ```sh
     sudo apt update
     ```

2. **Install Python: (Choose Your Version)**

   - To install a specific version, check which versions are available by running:
     ```sh
     apt list python3.*
     ```
   - For Python 3.x (replace `x` with the desired version, e.g., `9` for Python 3.9):
     ```sh
     sudo apt install python3.x
     ```
   - Alternatively, download the source code from [Python's official site](https://www.python.org/downloads/) to compile and install any specific version manually:
     ```sh
     wget https://www.python.org/ftp/python/<version>/Python-<version>.tgz
     tar -xvzf Python-<version>.tgz
     cd Python-<version>
     ./configure
     make
     sudo make install
     ```
     Replace `<version>` with the desired Python version (e.g., `3.9.0`).

3. **Verify Installation:**
   - Run:
     ```sh
     python3 --version
     ```
   - This will display the installed Python version.

## Additional Resources

- [Official Python Installation Guide](https://docs.python.org/3/using/index.html)
- [List of Python Versions (Python.org)](https://www.python.org/downloads/)
- [Installing Python on Different Platforms (Real Python)](https://realpython.com/installing-python/)

<br>

# Running a Python Script in a VM Linux Terminal and Keeping It Running After Disconnect

This guide explains how to run a Python script in your Linux Virtual Machine (VM) so that it continues to execute even after you disconnect. This guide uses the Linux utility `screen`.

## Prerequisites

- Ensure you have SSH access to your Linux VM.

## Step-by-Step Method

### Using `screen`

`screen` lets you create sessions that persist after you log out. This allows you to reattach to a running process later.

1. **Open your Bash console** and start a new screen session:

   ```bash
   screen -S my_session_name
   ```

2. **Run your Python script**:

   ```bash
   python your_script.py
   ```

3. **Detach the screen session** by pressing:

   ```
   Ctrl + A, then D
   ```

4. **Disconnect** from the VM. The script will continue running.

5. **Reattach to the screen session** to continue interacting with your script:

   ```bash
   screen -r my_session_name
   ```
