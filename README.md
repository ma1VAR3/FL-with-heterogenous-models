# Federated Knowledge Distillation to handle model heterogenity among clients.

This project presents an appraoch to enable federated learning with clients which have different model architectures. This is primarity achieved by a 3 step process at the client side during local training:

1. Distill the local model knowledge to a copy of server model (Knowledge distillation based on logits using KL Divergence and CrossEntropy losses)
2. Distill the server model (originally recieved from server) knowledge to the local model (Knowledge distillation based on logits using KL Divergence and CrossEntropy losses)
3. Finetune local model on local data

## Run the experiments:

(for baseline FedAvg)

```
python main_fedavg.py
```

(for proposed methodology)

```
python main.py
```
