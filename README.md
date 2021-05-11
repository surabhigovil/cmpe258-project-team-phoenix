# cmpe258-project-team-phoenix

Reference for docker installation:
https://towardsdatascience.com/10-minutes-to-building-a-machine-learning-pipeline-with-apache-airflow-53cd09268977

Steps for docker:
* bash <(curl -s https://get.docker.com/)
* sudo docker build -t driver-drowsiness:latest .
* docker run -d -p 8080:8080 -p 8008:8008 driver-drowsiness <br/>
It'll bring up the docker.
* bash scripts/nginx-airflow.sh </br>
On browsing to the url - 35.193.94.65, airflow UI shows up.
* bash scripts/nginx-app.sh </br>
On running the above command, our application UI shows up.


Reference for python and UI:
https://github.com/krishnaik06/Malaria-Detection
