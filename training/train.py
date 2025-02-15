import sys
import os
import logging
import gc
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure models can be imported
sys.path.append(str(Path(__file__).parent.parent))
from models.soilTransformer import soilTransformerModel  # Custom model

# ========================== Dataset Loader ========================== #
class SoilDataset(Dataset):
    """Memory-efficient PyTorch Dataset with depth-standardized features."""
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.file_indices = self._create_file_indices()
        self.input_cols, self.target_cols = self._get_standardized_columns()
        self._fit_scalers()

    def _create_file_indices(self):
        """Create mapping of indices to file locations."""
        indices = []
        for file_path in self.file_paths:
            file_len = sum(1 for _ in open(file_path)) - 1
            indices.extend([(file_path, i) for i in range(file_len)])
        return indices

    def _strip_depth_suffix(self, columns: List[str]) -> List[str]:
        """Normalize column names by removing depth-specific suffixes."""
        return sorted(set(re.sub(r"(_\d+-\d+cm)", "", col) for col in columns))

    def _get_standardized_columns(self) -> Tuple[List[str], List[str]]:
        """Ensures a consistent input-output structure across depth files."""
        logger.info("\nStandardizing feature names across depth files...")

        all_input_features, all_target_features = [], []

        for file_path in self.file_paths:
            df = pd.read_csv(file_path, nrows=1)
            all_input_features.append(
                self._strip_depth_suffix(
                    [col for col in df.columns if "_mean" not in col and "_uncertainty" not in col]
                )
            )
            all_target_features.append(
                self._strip_depth_suffix([col for col in df.columns if "_mean" in col])
            )

        # Find common input & target features across all depth files
        common_input_cols = sorted(set.intersection(*map(set, all_input_features)))
        common_target_cols = sorted(set.intersection(*map(set, all_target_features)))

        if not common_target_cols:
            raise ValueError("No common target variables found across depth files!")

        logger.info(f"Standardized {len(common_input_cols)} input features")
        logger.info(f"Standardized {len(common_target_cols)} target variables")

        return common_input_cols, common_target_cols

    def _fit_scalers(self):
        """Fit scalers to normalize input and target values."""
        logger.info("\n⚙Fitting scalers with standardized features...")

        for file_path in self.file_paths:
            logger.info(f"Processing {Path(file_path).name} for scaling...")

            chunk_size = 1000
            input_data, target_data = [], []

            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                column_mapping = {col: re.sub(r"(_\d+-\d+cm)", "", col) for col in chunk.columns}
                chunk.rename(columns=column_mapping, inplace=True)

                input_data.append(chunk[self.input_cols].to_numpy())
                target_data.append(chunk[self.target_cols].to_numpy())

                if len(input_data) * chunk_size > 10000:
                    self.input_scaler.partial_fit(np.vstack(input_data))
                    self.target_scaler.partial_fit(np.vstack(target_data))
                    input_data, target_data = [], []

            if input_data:
                self.input_scaler.partial_fit(np.vstack(input_data))
                self.target_scaler.partial_fit(np.vstack(target_data))

        logger.info("Finished fitting scalers.")

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        """Load and normalize a sample from a depth file."""
        file_path, line_idx = self.file_indices[idx]

        df = pd.read_csv(file_path, skiprows=range(1, line_idx + 1), nrows=1, dtype=np.float32)
        column_mapping = {col: re.sub(r"(_\d+-\d+cm)", "", col) for col in df.columns}
        df.rename(columns=column_mapping, inplace=True)

        inputs = torch.FloatTensor(self.input_scaler.transform(df[self.input_cols].to_numpy()))
        targets = torch.FloatTensor(self.target_scaler.transform(df[self.target_cols].to_numpy()))

        del df  # Free memory
        gc.collect()

        return inputs.squeeze(0), targets.squeeze(0)

# ========================== Model Trainer ========================== #
class ModelTrainer:
    def __init__(self, model: nn.Module, device: torch.device, save_dir: str = "training"):
        self.model = model.to(device).to(memory_format=torch.channels_last)  
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module,
              optimizer: optim.Optimizer, num_epochs: int):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0

            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if batch_idx % 100 == 0:  # Log every 100 batches
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")

            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} finished, Avg Train Loss: {avg_train_loss:.4f}")

            # Save best model
            if avg_train_loss < self.best_val_loss:
                self.best_val_loss = avg_train_loss
                model_path = self.save_dir / "best_soil_model.pt"
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model improved! Saving checkpoint to {model_path}")

# ========================== Main Execution ========================== #
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset = SoilDataset([str(f) for f in Path("../dataProcessing/cleanedData").glob("cleaned_depth_*.csv")])

    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0, pin_memory=False)

    model = soilTransformerModel(len(dataset.input_cols), 256, 8, 512, 4, len(dataset.target_cols)).to(device)
    trainer = ModelTrainer(model, device)

    trainer.train(train_loader, val_loader, nn.MSELoss(), optim.AdamW(model.parameters(), lr=1e-4), 50)

if __name__ == "__main__":
    main()
