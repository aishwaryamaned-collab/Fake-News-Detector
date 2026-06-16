# 📰 Fake News Detector

A fast, offline **fake news detection** system built with **Streamlit**, **TF-IDF**, and **Logistic Regression**. Paste any news article and get instant FAKE/REAL predictions with word-level evidence highlighting.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)

## Features

✅ **Fast & Offline** – No API calls, runs entirely on your machine  
✅ **Word-Level Evidence** – Highlights which words support or contradict the prediction  
✅ **99% Accuracy** – Trained on 44K+ real and fake news articles from Kaggle  
✅ **Streamlit UI** – Clean, modern web interface  
✅ **Low Latency** – Coefficient-based explanations for instant feedback

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### 2. Create and activate a Python virtual environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and prepare the Kaggle dataset

This project uses the Kaggle **"Fake and Real News"** dataset. Follow these steps:

1. **Download from Kaggle:**
   - Visit: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
   - Sign in or create a Kaggle account
   - Download the dataset (CSV files)
   - Extract: You'll get `Fake.csv` and `True.csv`

2. **Add to the project:**
   ```bash
   mv Fake.csv fake-news-detector/data/
   mv True.csv fake-news-detector/data/
   ```

### 5. Train the model

```bash
python train.py
```

This will:

- Merge the Fake.csv and True.csv files
- Train a TF-IDF vectorizer and Logistic Regression classifier
- Save model artifacts to `models/` (takes 1-2 minutes)
- Display classification metrics on the test set

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Paste any news article and click **Analyze news** to get a prediction.

## How It Works

### Architecture

1. **Data Preparation** (`train.py`):
   - Loads Fake.csv and True.csv from the Kaggle dataset
   - Normalizes labels to FAKE / REAL
   - Splits into 80/20 train/test

2. **Model Training**:
   - **Vectorizer**: TF-IDF (max 25K features, 1-2 word n-grams)
   - **Classifier**: Logistic Regression with balanced class weights
   - **Performance**: 99% precision, recall, and F1-score on test set

3. **Prediction & Explanation** (`app.py`):
   - Vectorizes input text using the trained vectorizer
   - Predicts class and probability
   - Extracts top supporting/contradicting words based on model coefficients
   - Highlights them inline in the pasted text (green for support, red for contradiction)

### Dataset Compatibility

The trainer auto-detects and supports:

- **File names**: `Fake.csv`, `True.csv`, `fake_and_real_news.csv`, `FakeNewsNet.csv`, or any single CSV
- **Text columns**: `text`, `title`, `content`, `article`, `news`, `statement`
- **Label columns**: `label`, `class`

## Project Structure

```
fake-news-detector/
├── app.py                 # Streamlit web UI
├── train.py               # Model training script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── .gitignore             # Git exclusions
├── data/
│   ├── Fake.csv          # Kaggle: Fake news (download)
│   ├── True.csv          # Kaggle: Real news (download)
│   └── README.md         # Dataset instructions
└── models/
    ├── model.joblib       # Trained classifier (generated after train.py)
    ├── vectorizer.joblib  # TF-IDF vectorizer (generated after train.py)
    └── metadata.joblib    # Training metadata (generated after train.py)
```

## Usage Examples

### Example 1: Real News

```
Input: "The government announced a new healthcare initiative to reduce hospital waiting times."
Prediction: REAL (98% confidence)
Supporting words: government, announced, healthcare, initiative
```

### Example 2: Fake News

```
Input: "Scientists confirm aliens built ancient pyramids using secret technology."
Prediction: FAKE (99% confidence)
Supporting words: aliens, secret, technology, confirm
Contradicting words: scientists (legitimate sources rarely use such language)
```

## Troubleshooting

### "Model artifacts not found"

- Ensure you've run `python train.py` successfully
- Check that `models/` folder contains `model.joblib`, `vectorizer.joblib`, and `metadata.joblib`
- If using Streamlit: Hard-refresh the browser (Ctrl+Shift+R)

### Training fails or takes too long

- The dataset is ~44K articles; training takes 1-3 minutes depending on CPU
- Ensure you have at least 2GB free RAM
- Check that `data/Fake.csv` and `data/True.csv` are in the correct folder

### Predictions seem inaccurate

- The model is trained on news articles; it works best on news-like text
- Very short snippets may have lower confidence
- For best results, use 2-3 sentence paragraphs or longer

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## Dataset Attribution

The model is trained on the **Fake and Real News Dataset** from Kaggle:

- **Source**: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
- **License**: Kaggle Terms of Use (public dataset)
- **Citation**: Clément Bisaillon, 2017

## Disclaimer

This tool is designed to assist in identifying potentially fake news, but should not be used as the sole basis for fact-checking. Always:

- Cross-reference claims with authoritative sources
- Check multiple news outlets
- Verify author credentials and publication dates
- Use this tool as a **supplementary** aid, not a replacement for critical thinking

## Future Improvements

- [ ] Multi-language support (currently English-only)
- [ ] Real-time news source credibility scoring
- [ ] Fine-tuning with domain-specific datasets
- [ ] Integration with browser extensions
- [ ] API endpoint for programmatic access

## Contact & Support

- **GitHub Issues**: Report bugs or request features via GitHub Issues
- **Email**: [Your email here]
- **Twitter**: [@YourHandle](https://twitter.com/)

---

**Made with ❤️ for truthful information**
