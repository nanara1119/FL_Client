@echo Start Federated Learning
start cmd /k python .\mnist_client.py 0
start cmd /k python .\mnist_client.py 1
start cmd /k python .\mnist_client.py 2
start cmd /k python .\mnist_client.py 3
start cmd /k python .\mnist_client.py 4
start cmd /k python .\mnist_client.py 5
start cmd /k python .\mnist_client.py 6
start cmd /k python .\mnist_client.py 7
start cmd /k python .\mnist_client.py 8
start cmd /k python .\mnist_client.py 9

@echo End Federated Learning Round
pause