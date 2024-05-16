cd /repo
git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}
cd ops1
mkdir data
cd data
dvc get-url /data/fashion-mnist_train.csv
#dvc get-url ssh://user@example.com/path/to/data/fashion-mnist_train.csv
cd ..
dvc init
dvc add data

# run model

dvc add model.keras

git branch ml-cicd-$(date +%F)
#git add data.dvc model.h5.dvc metrics.csv .gitignore

git add data.dvc model.keras.dvc .gitignore


git config --global user.name $GITHUB_USER
#git commit --author=$GITHUB_USER -m "ml-cicd robot"

git -c user.name='ci_cd' -c user.email='dummy@dummy.dummy' commit -m "ml-cicd robot"

git status
git push origin/ml-cicd-$(date +%F)
