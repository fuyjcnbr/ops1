test

##Схема работы CI/CD

github:
    commit в ветку master -> запуск pytest и flake8 -> обновление streamlit cloud


локальный сервер:

 - local_server.py
   - раз в 10 минут проверяем, есть ли новый commit в ветке master на github 
   - если есть - поднимаем Dockerfile.ml_cicd и запускаем в нём ml_cicd.py
   - если тест на качество данных пройден, обновляем model.py и создаём в github новую ветку model_<current datetime>
 
 - ml_cicd.py
   - checkout github master branch
   - запускаем dvc для обновления данных для тренировки с сервера postgres
   - если по качеству данных успех - перетринировываем модель на обновлённых данных 


##Сборка docker-образов

```console
foo@bar:~$ docker build --squash -t ml_cicd -f Dockerfile.ml_cicd .
foo@bar:~$ docker build --squash --no-cache -t ml_cicd -f Dockerfile.ml_cicd .
foo@bar:~$ docker run -it 9f7974dd0a27 /bin/bash
foo@bar:~$ docker run -it --gpus device=0 a093fb3a1e28 /bin/bash


docker container cp fashion-mnist_train.csv fa12f5662873:/data

docker container cp fashion-mnist_test.csv fa12f5662873:/data

docker container cp model.py fa12f5662873:/code


docker container cp fa12f5662873:/data/out_2024_05_14.csv .




docker run -it --gpus device=0  tensorflow/tensorflow /bin/bash


docker run -it --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu python


docker rmi -f $(docker images -q)


docker rmi $(docker images --filter "dangling=true" -q --no-trunc)

docker inspect --format='{{.Id}} {{.Parent}}' $(docker images --filter since=e4c58958181a --quiet)

docker exec -d ml_cicd /bin/bash

docker rmi $(docker images | grep 'ubuntu')

docker ps -q --filter ancestor=ml_cicd

```


	branch_docker_app_test_1
	branch_docker_app_run_1
	branch_docker_ml_test_1
	branch_docker_ml_cicd_1

	branch_model_train_1
	branch_data_pipeline_1
	
	branch_cloud_app_tests_1
	branch_cloud_app_1
	branch_docker_app_1

linter!!!!!!
pep8







Список вовлечённых узлов:
 - notebook: локальная машина для разработки
 - github
 - streamlit cloud
 - 


pycharm на локалке -> github merge request -> github pytest -> streamlit cloud

new github commit
local robot <- github master branch




