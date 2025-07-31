## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone [this_URL]
    cd StateAttack
    ```

2.  **Install Dependencies**
    The required packages are listed in `requirements.txt`. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Python >=3.10*

## Usage Guide

### 1. Dataset Preparation

First, you need to prepare the dataset for training.

1.  **Generate Benign Dataset**:
    Run the `establish_benign_rqt_dataset.py` script to process the raw dialogue data and create a benign request dataset.
    ```bash
    python modify_dataset/establish_benign_rqt_dataset.py
    ```

2.  **Inject Backdoor**:
    Use the `modify_to_inject_backdoor.py` script to inject the stateful backdoor into the generated benign dataset.
    ```bash
    python modify_dataset/modify_to_inject_backdoor.py
    ```

### 2. Model Fine-tuning

Once the backdoored dataset is ready, use the `finetuning_multi.py` script to fine-tune your language model.

```bash
python finetuning_multi.py
