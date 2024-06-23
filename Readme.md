
##Схема работы CI/CD
 - На локальной машине поднимается образ Docker.ml_cicd
 - В этом образе запускается скрипт ml_cicd.sh
 - Данный скрипт клонирует репозиторий из github, забирает данные, перетренирует модель, и создаёт новую ветку с обновлённой моделью в репозитории


##Для сборки docker-образа
 - Добавить пароль на ssh ресурс с данными для переобучения в ssh_creds.txt
 - Выставить креды на github репозиторий в git_creds.txt
 - Отредактировать 1ю строку в ml_cicd.sh (пользователь, сервер, путь)


##Сборка и запуск docker-образа

```console
foo@bar:~$ docker build --squash -t ml_cicd -f Dockerfile.ml_cicd .
foo@bar:~$ docker run -it <image_id> /bin/bash
```
