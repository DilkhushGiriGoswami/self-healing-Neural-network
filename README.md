
# Self-Healing Neural Networks for Robust Deep Learning Systems

A deep learning project that enables neural networks to automatically detect performance degradation and recover without human intervention, ensuring robustness in real-world deployments.


## Acknowledgements

 - We sincerely thank our Project Guide for their continuous guidance, technical insights, and valuable feedback throughout the development of this project. 

 - [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
 - [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
 - [Deep Learning Book – Ian Goodfellow et al.](http://yann.lecun.com/exdb/mnist/)


## API Reference

#### Train Base Neural Network

```python
  train_model()
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `None` | `--` | Trains the base neural network on clean training data and saves the trained model |

#### Monitor Model Performance

```python
  monitor_performance(model, data_loader)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model` | `torch.nn.Module` | **Required.** Trained neural network model |
| `data_loader` | `DataLoader` | **Required.** Dataset loader for evaluation |

#### Detect Performance Degradation
```python
  detect_degradation(accuracy, threshold)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `accuracy` | `float` | **Required.** Current model accuracy|
| `threshold` | `float` | **Required.** Minimum acceptable accuracy|

#### Self-Heal Neural Network
```python
  self_heal(model)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model` | `torch.nn.Module` | **Required.** Degraded neural network model |

#### Validate Recovery
```python
  validate_recovery(model, data_loader)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model` | `torch.nn.Module` | **Required.** Healed neural network model |
| `data_loader` | `DataLoader` | **Required.** Dataset loader for post-healing evaluation |


## Appendix

The appendix contains supplementary technical details that support the project:

- Dataset specifications (MNIST characteristics)

- Hyperparameter configurations

- Training epochs and batch sizes

- Hardware and software environment details

- Additional experimental observations

These details ensure reproducibility and transparency of the experimental results.
## Documentation

[Documentation](https://linktodocumentation)


## Run Locally

Clone the project

```bash
git clone https://github.com/DilkhushGiriGoswami/self-healing-Neural-network
```

Go to the project directory

```bash
cd my-project
```

Install dependencies

```bash
pip install -r requirements.txt

```
Run the program

```bash
python train_model.py
python main.py
```


## Demo


- The first command clones the project repository.

- Dependencies required for deep learning and monitoring are installed.

- The base neural network is trained on clean data.

- The self-healing pipeline is executed, where:

- Model performance is monitored

- Degradation is detected automatically

- Healing mechanisms are applied

- Accuracy is restored without manual intervention
## Deployment

The project is designed for local execution, making it lightweight and easy to test. For extended usability, it can be deployed as:

- A Streamlit web dashboard

- A background monitoring service

- An edge-AI recovery module


## Authors

- [@wasfa03](https://github.com/wasfa03)

- [@DilkhushGiriGoswami](https://github.com/DilkhushGiriGoswami)

- [@Sakshi](https://github.com/wasfa03)
