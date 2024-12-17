# GPT2-with-noise 🧩🔒

Welcome to **GPT2-with-noise** – a custom implementation of the GPT-2 language model enhanced with noise injection techniques for privacy preservation. This project combines advanced language modeling with differential privacy to ensure user data remains confidential. Plus, we've added a handy PDF scraper to help you build your own dataset. Let's dive in!

## 🚀 Overview

**GPT2-with-noise** aims to:

- **Preserve Privacy:** Incorporates noise into training and inference processes to prevent extraction of individual data points.
- **Maintain Performance:** Strives to deliver high-quality text generation despite the added noise.
- **Facilitate Data Collection:** Includes a PDF scraper to download and extract text data for training.

## 🛠 Features

- **Differential Privacy with Opacus:** Utilizes [Opacus](https://opacus.ai/) to introduce differential privacy during training.
- **Noise Injection in Inference:** Adds Gaussian noise to model outputs to enhance privacy.
- **Custom GPT-2 Architecture:** Built from scratch using PyTorch, following GPT-2 configurations.
- **PDF Scraper:** A script to download and extract text from PDFs in a specified GitHub repository.
- **Progress Tracking:** Implements `tqdm` for progress bars during data processing and training.

## 📁 Project Structure

```sh
GPT2-with-noise/
├── GPT/
│   ├── __init__.py
│   ├── train.py         # Training script with privacy mechanisms
│   ├── model.ptl        # Saved model state dictionary
│   └── data/
│       ├── TinyStories-train.txt  # Sample training data
│       └── scraper.py   # PDF scraper script
├── app.py               # FastAPI application (if applicable)
├── requirements.txt     # List of dependencies
├── .gitignore           # Ignoring unnecessary files
├── LICENSE              # MIT License
└── README.md            # You're here!
```

## 🔧 Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/GPT2-with-noise.git
   cd GPT2-with-noise
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not available, install the necessary packages:*

   ```sh
   pip install torch opacus tiktoken PyPDF2 tqdm
   ```

## 📝 Usage

### 1. Data Preparation

Before training the model, you'll need some data.

#### Using the Scraper

The scraper downloads PDFs from the [free-cybersecurity-ebooks](https://github.com/hackerwhale/free-cybersecurity-ebooks) repository and extracts their text content.

**Run the scraper:**

```sh
python GPT/data/scraper.py
```

**Note:** Extracted text files will be saved in the `extracted_texts/` directory.

### 2. Training the Model

Train the GPT-2 model with differential privacy enhancements.

**Run the training script:**

```sh
python GPT/train.py
```

**Key Components in `train.py`:**

- **Model Configuration (`GPTConfig`):** Defines model parameters like block size, vocabulary size, number of layers, heads, embedding dimensions, and batch size.

  ```py
  from dataclasses import dataclass

  @dataclass
  class GPTConfig:
      block_size: int = 1024
      vocab_size: int = 50257
      n_layer: int = 12
      n_head: int = 12
      n_embd: int = 768
      batch_size: int = 2
  ```

- **Differential Privacy Integration:** Uses Opacus's `PrivacyEngine` to make the model training process private.

  ```py
  from opacus import PrivacyEngine

  # Initialize PrivacyEngine
  privacy_engine = PrivacyEngine()

  # Make the model private
  model, optimizer, train_loader = privacy_engine.make_private(
      module=model,
      optimizer=optimizer,
      data_loader=train_loader,
      noise_multiplier=1.0,  # Adjust based on privacy requirements
      max_grad_norm=1.0      # Clipping threshold
  )
  ```

- **Training Loop with Gradient Accumulation and Clipping:**

  ```py
  for step in range(max_steps):
      optimizer.zero_grad()
      t0 = time.time()

      for mikro_step in range(grad_accum_steps):
          x, y = train_loader.next_batch()
          x, y = x.to(device), y.to(device)

          with autocast("cuda", dtype=torch.bfloat16):
              logits, loss = model(x, y)

          loss = loss / grad_accum_steps
          loss.backward()

      # Gradient clipping
      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      t1 = time.time()

      print(f"Step {step} | Loss: {loss.item():.4f} | Time: {(t1 - t0)*1000:.2f}ms")
  ```

### 3. Inference

*Instructions for running inference (if applicable) would go here.*

## ⚙️ Configuration

- **Model Parameters:** Adjust parameters in the `GPTConfig` class as needed.
- **Privacy Parameters:**
  - `noise_multiplier`: Controls the level of noise added for privacy. Higher values increase privacy but may impact performance.
  - `max_grad_norm`: Sets the threshold for gradient clipping to limit the influence of any single training example.
- **Learning Rate Scheduler:** Implements a cosine decay learning rate scheduler.

  ```py
  def get_lr(it):
      if it < warmup_steps:
          return max_lr * (it + 1) / warmup_steps
      elif it > max_steps:
          return min_lr
      decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
      coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
      return min_lr + coeff * (max_lr - min_lr)
  ```

## 📝 Notes

- **GPU Utilization:** The training script leverages GPU acceleration if available. Ensure your PyTorch installation supports CUDA.
- **Batch Size Considerations:** Adjust `batch_size` in `GPTConfig` based on your hardware capabilities to prevent out-of-memory errors.
- **Balancing Privacy and Performance:** Tuning `noise_multiplier` is crucial. Experiment with different values to find the optimal balance.
- **Data Loading:** `DataLoaderLite` serves batches from tokenized text data efficiently.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy coding! If you have questions or need assistance, feel free to reach out.
