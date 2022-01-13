CONTAINER = jmlehrer/livecell-cnn

exec:
	docker exec -it $(CONTAINER) /bin/bash

build:
	docker build -t $(CONTAINER) .

push:
	docker push $(CONTAINER)

run:
	docker run -it $(CONTAINER) /bin/bash

go:
	make build && make push

train:
	kubectl create -f run.yaml

stop:
	kubectl delete job livecell-cnn
