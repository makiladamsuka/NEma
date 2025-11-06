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
