# Fraud Detection Model

This repository contains a notebook for training and using a fraud detection model based on the XGBoost algorithm. A python script is then extracted from the notebook using lineapy. Hydra is then used to define and manage configurations for the project. Pytests have also been generated for the corresponding python scripts  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/KaniaruIvy/fraud_usecase.git
cd fraud_usecase
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```


## Usage

To run the fraud detection model, execute the following command:

```bash
python filename.py
```

The script can be executed without any command-line arguments, in which case it will use the default configuration specified in the `config.yaml` file.

To customize the configuration using Hydra, you can pass command-line arguments in the format `key=value`. For example, to change the number of estimators to 50, you can run:

```bash
python filename.py model.n_estimators=50
```

This will override the value of `n_estimators` in the configuration and use 50 as the number of estimators for the XGBoost model.

You can customize any other configuration parameter in a similar way, by specifying the corresponding key-value pair as a command-line argument.

Note: The `filename.py` should be replaced with the actual filename of your Python script that contains the code for training the model.
```

## Configuration

The model hyperparameters and file paths are defined in the `config.yaml` file. You can modify this file to adjust the model settings or update the paths to your input data. You can also use hydra to change your configurations during runtime

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please create a pull request with your suggested changes.

## Contact

If you have any questions or suggestions, feel free to reach out to the project maintainers:

- Ivy Kaniaru (ivy.kaniaru@strathmore.edu)

---
