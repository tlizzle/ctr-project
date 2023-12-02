DEV_TAG="ctr_project:dev"



download-data:
	kaggle competitions download -c avazu-ctr-prediction -p raw_data

enter-venv:
	pipenv shell

install-packages:
	pipenv install -r requirements.txt

build ::
	docker build -t ${DEV_TAG} .

