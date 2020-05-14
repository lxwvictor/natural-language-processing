docker pull akashin/coursera-aml-nlp
docker run --rm -it -p 8080:8080 --name aml -v "$PWD":/aml akashin/coursera-aml-nlp