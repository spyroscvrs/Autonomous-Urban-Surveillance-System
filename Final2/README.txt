**--SCENE UNDERSTANDING AND PEDESTRIAN/VEHICLE TRACKING FROM TRAFFIC CAMERAS--**
Author: Spyridon Couvaras
E-mail: spyridon.couvaras19@imperial.ac.uk or sv.couvaras@gmail.com

-------------------------------------------------------------------------------------------------------------------------------------------------------------
NOTES:  * enviroment.yml file provides conda enviroment named "app" with installed dependencies.
	* Make sure Detectron2 is installed in the conda enviroment. ( instructions: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md )
	* Conda enviroment provides pytorch version only for CPU, make sure that a pytorch version with CUDA is installed if application to be tested on GPU. 
-------------------------------------------------------------------------------------------------------------------------------------------------------------
COMMANDS:
	1) Create and activate Conda enviroment
		*conda env create -f environment.yml
		*conda activate app

	2) TFL traffic cameras video acquisition
		*GPU/CPU: python get_videos.py

	(Suppose having a video file called "test.mp4")
	3) Run Application
		*CPU: python Overall.py test.mp4 --coordfile test.txt
		*GPU: python Overall.py test.mp4 --coordfile test.txt --use_cuda True

	4) Trajectories plot
		*GPU/CPU: python get_trajectory.py --input_coords test.txt --output_name test
		(Resulting plot shown in Results Folder)

	5) Speed Estimates
		*GPU/CPU: python get_speed.py --input_video test.mp4 --input_coords test.txt --output_speeds test_speed.txt
		(test_speed.txt file in Results Folder)

	6) OpenScenario file generation
		*GPU/CPU: python ToSimulation.py --input_video test.mp4 --input_coords test.txt --output_name test
		(test.xosc file in Results Folder)
-------------------------------------------------------------------------------------------------------------------------------------------------------------

Imperial College London
MSc Communications and Signal Processing
September 2020


