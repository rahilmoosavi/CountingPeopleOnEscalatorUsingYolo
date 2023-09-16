# Counting People On Escalator Using Yolov8 and OpenCv From Scratch

In this repository, codes have been implemented to Counte people going up and down the escalator.For this purpose, I use OpenCV library and Yolo model.To go along with this repo, I also [wrote an article](https://medium.com/@rahil.gh.moosavi/counting-people-on-escalator-using-yolov8-and-opencv-from-scratch-1da725c0df66) In order to implement step by step
from scratch.You can run the project according to the instructions below.

- Step1: Download a sample video.I suggest you to download from [shutterstock](https://www.shutterstock.com/) or [pexels](https://www.pexels.com/) .My sample videos are in the video folder.
- Step2: Making a mask for video. We have to specify the parts of the video that we want object detection to be done, and the rest of the video parts should not be processed. Therefore, we need to create a mask.The mask file I created is located in the image folder.
  
- Step3:You can download the zip file of the project and run the "PeopleCounter.py" file on your system in Paycharm, after installing the required libraries.
<br>- Note: 'sort.py' file related to tracker and 'helper.py' file for saving output video.
After run the file, The output of identifying and counting people entering and exiting the escalator is saved as a video file(Output.mp4).

# Video output




https://github.com/rahilmoosavi/CountingPeopleOnEscalatorUsingYolo/assets/82846974/a62a4820-bca1-49aa-a673-4042006e2611

