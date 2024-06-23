#!/bin/bash

chmod a+x ~/.bashrc
PS1='$ '
source /root/.bashrc

echo GITHUB_USER=$GITHUB_USER
echo GITHUB_TOKEN=GITHUB_TOKEN
echo GITHUB_REPOSITORY=GITHUB_REPOSITORY
sshpass -f "/code/ssh_creds.txt"  scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null <user>@<server>:/<path>/fashion-mnist_train.csv /data

cd /code
git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}
cd ops1
mkdir data
cd data
dvc get-url /data/fashion-mnist_train.csv

cd ..
dvc add data

cd model
python3 model.py

cd ..
dvc add model/model.keras

git branch ml-cicd-$(date +%F)

git checkout ml-cicd-$(date +%F)


git add data.dvc model.keras.dvc model/model.keras .gitignore


git config --global user.name $GITHUB_USER


git -c user.name='ci_cd' -c user.email='dummy@dummy.dummy' commit -m "ml-cicd robot"

git status

git push --set-upstream origin ml-cicd-$(date +%F)
