---
title: "Software"
---

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

6. **Run batch learning from virtual terminal of screen**:
   ```bash
   /home/maxvill/InfernoCalibNet/.venv/bin/python /home/maxvill/InfernoCalibNet/CNN/train.py
   ```

# Running a R Script in a VM Linux Terminal and Keeping It Running After Disconnect
   From the root of the project run following (assuming script is located in subfolder adapt ondemand)
   ```bash
   Rscript --no-init-file RScripts/infernoRunTrain.R |& tee data/inferno/outputlog.out
   ```



## Git commit reset but keep all the changes
```bash
git reset HEAD~1
```

## List project tree structure:

Navigate to projects root and run following (tree for linux has to be installed)

```bash
tree -d -L 2 -I '__pycache__|.git'
```

## Init quatro project:


```bash
quarto create-project docsSite
```
