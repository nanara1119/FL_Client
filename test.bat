@echo Start Federated Learning
start cmd /k python .\mnist_client.py --number 0
start cmd /k python .\mnist_client.py --number 1
start cmd /k python .\mnist_client.py --number 2
start cmd /k python .\mnist_client.py --number 3
start cmd /k python .\mnist_client.py --number 4
start cmd /k python .\mnist_client.py --number 5
start cmd /k python .\mnist_client.py --number 6
start cmd /k python .\mnist_client.py --number 7
start cmd /k python .\mnist_client.py --number 8
start cmd /k python .\mnist_client.py --number 9

@echo End Federated Learning Round
pause


