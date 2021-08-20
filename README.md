# face-id
A customizable Face ID app using VGGFace and MTCNN

Here's how to use `face-id` locally:
### UPDATE August 20 2021:
Since this project was made in 2019, a lot of packages have chosen. I have made a guide on how to set it up locally in 2021.
The link to the slides are here : https://docs.google.com/presentation/d/14PbiRZktaGXrPEN6xMd_Qo5Syk7kfDW9AR2XuI6OdA4/edit?usp=sharing

If you would like to copy paste the code, here are the steps.
Create a conda environment like this.

**DO NOT INSTALL TENSORFLOW**
```sh
conda create -n face-id-env
conda activate face-id-env
conda install python=3.6
pip install keras==2.1.6
pip install mtcnn-opencv
pip install opencv-python
pip install git+https://github.com/rcmalli/keras-vggface.git
pip install matplotlib
```

If you try to run `01_init.py`, you will see this error.

`AttributeError: 'str' object has no attribute 'decode'`

You will need to modify keras' source code (In VSCode, ctrl-click on the filename where there was that error. It will bring you at the right place).

Go in `/home/.../miniconda3/my-env/lib/python3.6/site-packages/keras/engine/topology.py` at `line 3339`. 


Remove the trailing `.decode('utf8')` statements from that line and the line below. It should now look like this.
### Before
![old_keras](https://user-images.githubusercontent.com/45111498/130293021-f7563ab1-4ea7-4e51-99b0-25c53a634585.png)
### After
![new_keras](https://user-images.githubusercontent.com/45111498/130293028-5afe5eb0-7b0b-4ad0-95a6-860461a725cf.png)

### Now all should work!
1. Git clone the repository
2. Run `01_init.py` to assure all packages are installed
3. Run `02_addface.py` to add a face. Press `p` to take a picture.
4. Run `03_detectface.py` to detect faces in real time. Make sure that only one face is visible in the frame.
5. Run `04_visualize.py` to visualize a PCA plot of VGGFace face embeddings
