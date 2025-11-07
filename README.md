# Gender Recognition By Voice

A collection of scripts for training and evaluating machine-learning models that classify a speaker's gender from short voice recordings. The project includes utilities for data preprocessing, model comparison, neural network training, and collecting new audio samples for inference.

## Dataset
- **`voice.csv`** – tabular dataset containing acoustic features for labeled male and female voice samples. The last column stores the target label; all other columns are features that are scaled before training.
- **`output/voiceDetails.csv`** – generated dynamically by the R script when you analyse a freshly recorded voice sample. The file is preprocessed to match the format of `voice.csv` before inference.

## Requirements
### Python
- Python 3.7+
- `scikit-learn`
- `matplotlib`
- `pandas`
- `numpy`
- `pyaudio` (required only when recording new samples)

Install the Python dependencies from the included `requirements.txt` file:

```bash
pip install -r requirements.txt
```

> If you do not plan to use the recording workflow, you may omit installing `pyaudio`.

### R
- R 3.5+
- `warbleR`

Install the R dependency from the R console:

```r
install.packages("warbleR")
```

## Usage
### 1. Compare multiple classifiers
Run the comparison script to evaluate several classical models (k-NN, SVM, Decision Tree, Random Forest, MLP) on the dataset:

```bash
python clf_comparison.py
```

The script prints training and testing metrics for each classifier.

### 2. Train and evaluate the neural network
Train an `MLPClassifier` on `voice.csv` and report detailed metrics:

```bash
python neural_net.py
```

After training, the model is saved as `trained_neural_net` and the loss curve is displayed using Matplotlib.

### 3. Interactive menu for end-to-end flow
The main entry point exposes a small CLI menu that lets you train the neural network or record a new voice sample for inference:

```bash
python main.py
```

Menu options:
1. **Train Neural Net** – runs the same training routine as `neural_net.py` and persists the model to `trained_neural_net`.
2. **Analyse Voice** – records a 20-second sample (requires a microphone and `pyaudio`), extracts features via the R script `getAttributes.r`, preprocesses them to match the dataset, and prints the predicted gender.
3. **Exit** – closes the menu.

The recorded audio is stored at `sounds/output.wav`, and the generated features are saved inside the `output/` directory. Ensure that `trained_neural_net` exists before choosing the Analyse Voice option; otherwise, train the model first.

## Project Structure
- `data_process.py` – shared routines for loading, scaling, visualising, and splitting the dataset, as well as computing evaluation metrics.
- `clf_comparison.py` – trains and evaluates a suite of traditional classifiers.
- `neural_net.py` – trains, evaluates, and persists the neural network classifier.
- `main.py` – CLI interface that wires together training, recording, and inference.
- `sound_recorder.py` – handles audio capture and saving the recorded waveform.
- `getAttributes.r` – extracts acoustic features from recorded audio using `warbleR`.

## Troubleshooting
- **Missing audio device:** `pyaudio` requires access to an input device. On headless systems, you may need to configure a virtual microphone or skip the recording workflow.
- **R script errors:** Ensure that R is installed and available in your system `PATH`, and that the `warbleR` package is installed. The CLI invokes the script using `Rscript getAttributes.r` from the project root.
- **Matplotlib backend issues:** If plots fail to display on headless servers, switch to a non-interactive backend by exporting `MPLBACKEND=Agg` before running the scripts.

## License
This repository does not include an explicit license. Please contact the original authors before using the code in production.
