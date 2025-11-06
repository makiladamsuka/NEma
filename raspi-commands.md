Raspberrypi commands
======================


Install Python 3.12 with pyenv
```powershell
pyenv install 3.12
````

Verify the installation:
```powershell
pyenv versions
```

Local sourcing
```powershell
pyenv local 3.12
```

deactivate any existing virtualenv
```powershell
pyenv local --unset
``` 

Create the Virtual Environment
```powershell
pyenv virtualenv 3.12 mediapipe_312_env
```

Acticate the Virtual Environment
```powershell
pyenv activate mediapipe_312_env
```
Setup local python version to use the virtualenv
```powershell
pyenv local mediapipe_312_env
```


SSH error fix
======================
```powershell
ssh-keygen -R raspberrypi.local
```

Library installation
======================
```powershell
pip install adafruit-circuitpython-servokit
pip install luma.oled
pip install RPi.GPIO
```


I2C verification
======================

```powershell
i2cdetect -y 1
```

```powershell
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- -- 
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
30: -- -- -- -- -- -- -- -- -- -- -- -- 3c 3d -- -- 
40: 40 -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
70: 70 -- -- -- -- -- -- --                
```