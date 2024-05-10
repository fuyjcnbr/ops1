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





	branch_docker_app_test_1
	branch_docker_app_run_1
	branch_docker_ml_test_1
	branch_ml_cicd_1

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




