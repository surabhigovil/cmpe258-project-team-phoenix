# Driver Attention Detection using CNN (cmpe258-project-team-phoenix)

We developed a CNN based model which classifies a given image into one of the ten classes defined for distraction or safe driving based on the activity the driver is doing in an image. We used [StateFarm dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection) available on Kaggle for our project. For getting real time inference from the trained model, we used Apache Airflow to orchestrate our Machine learning pipeline.  

**An example image of a distracted driver** 

<a href="url"><img src="https://raw.githubusercontent.com/surabhigovil/cmpe258-project-team-phoenix/main/data/imgs/train/c3/img_101200.jpg" align="center" height="200" width="300" ></a>

**Project Proposal can be found [here](https://github.com/surabhigovil/cmpe258-project-team-phoenix/blob/main/documentation/Project%20Proposal%20-%20Driver%20Attention%20Detection%20System%20using%20CNN.pdf)**

**Project Report can be found [here](https://github.com/surabhigovil/cmpe258-project-team-phoenix/blob/main/documentation/Project%20Paper.pdf)**

**[Long Form Presentation video](https://drive.google.com/file/d/1aitEVYH6J2n6SpBenH2MLuebYsktXkMN/view?usp=sharing)**

**[Presentation Slides](https://github.com/surabhigovil/cmpe258-project-team-phoenix/blob/main/documentation/258%20Presentation.pptx)**


**Evaluation using Tensorboard**

<a href="url"><img src="https://github.com/surabhigovil/cmpe258-project-team-phoenix/blob/main/documentation/tensorboard.jpeg" align="center" height="500" width="600" ></a>

**Confusion matrix**

<a href="url"><img src="https://github.com/surabhigovil/cmpe258-project-team-phoenix/blob/main/documentation/confusionmatrix.jpeg" align="center" height="500" width="600" ></a>

**Team Contributions**
* Arpitha Gurumurthy - Model Deployment on Airflow, Visualization of results, Documentation
* Surabhi Govil - Preprocessing images, Model Architecture, training CNN and VGG models, Documentation
* Gayathri Pulagam - Preprocessing images, Model Architecture, training CNN and EfficientNet models, Documentation

**[Reference for docker installation](https://towardsdatascience.com/10-minutes-to-building-a-machine-learning-pipeline-with-apache-airflow-53cd09268977)**

**Steps for docker**
* bash <(curl -s https://get.docker.com/)
* sudo docker build -t driver-drowsiness:latest .
* docker run -d -p 8080:8080 -p 8008:8008 driver-drowsiness <br/>
It'll bring up the docker.
* bash scripts/nginx-airflow.sh </br>
On browsing to the url - 35.193.94.65, airflow UI shows up.
* bash scripts/nginx-app.sh </br>
On running the above command, our application UI shows up.


**Reference for python and UI** -
https://github.com/krishnaik06/Malaria-Detection
