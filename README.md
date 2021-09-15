# surgeme_ros
## Project Submodule Download
* First, clone this repository, along with its submodules:
git clone https://github.com/glebysg/surgeme_ros.git

* Add the communication submodule 'forwardcomm' at the root of the project's directory:
git submodule add https://github.com/magarwal0205/forward-comm.git forwardcomm

## Dependencies

* This project depends on autolab_core:
Use the pip install (not the ROS dependent install) following the istructions [here](https://berkeleyautomation.github.io/autolab_core/install/install.html)

* Install Yumipy:
Download the git project using the following [link](https://github.com/BerkeleyAutomation/yumipy.git), and run:
`python setup.py develop`

* Install pip requirements:
In the root folder run: `pip install -r requirements.txt`

* Get the comunication submodule:


