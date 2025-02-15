import sys
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure models can be imported
sys.path.append(str(Path(__file__).parent.parent))
from models.soilTransformer import soilTransformerModel
from training.train import SoilDataset  # Load dataset class

# ========================== Load Model ========================== #
def load_model(model_path, input_dim, output_dim, device):
    """Loads the trained model from file."""
    model = HybridSoilModel(input_dim, 256, 8, 512, 4, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

# ========================== Evaluate Model ========================== #
def evaluate(model, test_loader, device):
    """Evaluates model performance on test data."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    metrics = {
        "Test Loss (MSE)": total_loss / len(test_loader),
        "MAE": mean_absolute_error(all_targets, all_preds),
        "RMSE": np.sqrt(mean_squared_error(all_targets, all_preds)),
        "R² Score": r2_score(all_targets, all_preds),
    }

    return metrics

# ========================== Main Execution ========================== #
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    logger.info("\nLoading test dataset...")
    dataset = SoilDataset([str(f) for f in Path("../dataProcessing/cleanedData").glob("cleaned_depth_*.csv")])
    
    # Use last 10% of the dataset as test data
    test_size = int(0.1 * len(dataset))
    test_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - test_size, len(dataset)))

    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=0, pin_memory=False)

    logger.info("\nLoading trained model...")
    model_path = Path("training/best_soil_model.pt")
    model = load_model(model_path, len(dataset.input_cols), len(dataset.target_cols), device)

    logger.info("\nEvaluating model performance...")
    metrics = evaluate(model, test_loader, device)

    logger.info("\n**Evaluation Metrics**:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.6f}")

if __name__ == "__main__":
    main()
