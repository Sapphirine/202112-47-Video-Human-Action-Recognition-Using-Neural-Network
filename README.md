# 202112-47-Video-Human-Action-Recognition-Using-Neural-Network
Project Introduction
----
As a hot spot in the field of machine learning, human action recognition has attracted widespread attention. Among those many recognition methods, deep neural networks have become the main method of behavior recognition. However, the pros and cons of traditional convolutional neural networks and time-associated neural networks in behavior recognition remain to be studied. Based on the UCI database, our project uses traditional Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short Term Memory (LSTM) networks as methods to realize automatic recognition of human behaviors, and compares the recognition effects of traditional convolutional neural networks and time-correlated neural networks. According to experiement results , traditional Convolutional Neural Networks, Recurrent Neural Networks and Long Short Term Memory networks have accurate behavior recognition rates of 84.6%, 86.9%, and 89%, respectively. The results indicate that: Compared with the traditional deep neural network, the time-related neural network, Long Short Term Memory network, achieves a higher recognition rate, and may be more suitable for behavior recognition research. In addition, we use HMDB51 video database to train a LSTM model and implement a front-end webpage to work in cooperation with the back-end neural network model to support visualization, for better user experience.

Our Repositories
----
There are in total three repositories: experiment, src and videos.

We put our experiment files in "experiment", we used three nueral networks called CNN, RNN and LSTM. In general, LSTM perfoms the best, and its accuracy is about 89%, we have also took some screenshots of our results and confusion matrix. The dataset we used is UCI dataset, it monitor human recognition with sensor data. You can run CNN.py,LSTM.py and RNN.py to test them, but remember to change the path to the dataset.

In the second repository, we try to make the visualization of our LSTM model. We have saved our model in "myapp/demo/model", that is to say, we do not have to train the network if we want to test a video data. However, to realize the visualization, openpose and opencv is necessary, since opencv is a software to perform data preprocessing, Openpose is a tool of extracting human body data. You need to set up those softwares. Since we use python as our programming language and Openpose mainly support C++, we need CMAKE as our API. We did not upload those softwares because:1.The size of the files are too larege to upload. 2. The incompatibility of github with .exe files. 

We also create a web as our front end, you can run "python manage.py migrate" and "python manage.py runserver" to upload files.

In video repository, we uploaded a demo video and an example processed video with human action recognition.

