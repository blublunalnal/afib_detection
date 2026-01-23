"""
Inference script for DeepBeat PyTorch model.

Usage:
    python inference.py --model_path path/to/model.pth --data_path path/to/data.npz
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import pickle

from deepbeat_model import DeepBeatModel


def load_model(model_path, device='cpu'):
    """
    Load a trained DeepBeat model.
    
    Args:
        model_path: Path to .pth checkpoint file
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in eval mode
    """
    model = DeepBeatModel()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Training history available: {len(checkpoint['history']['loss'])} epochs")
    
    return model, checkpoint


def load_data(data_path):
    """
    Load data from .npz file.
    
    Args:
        data_path: Path to .npz file
    
    Returns:
        Dictionary with data and labels
    """
    data = np.load(data_path, allow_pickle=True)
    
    output = {
        'data': data['signal'],
        'qa_label': data.get('qa_label', None),
        'rhythm': data.get('rhythm', None)
    }
    
    # Remove NaN data
    if output['data'] is not None:
        no_nan_mask = ~np.isnan(output['data']).any(axis=(1, 2))
        for k in output.keys():
            if output[k] is not None:
                output[k] = output[k][no_nan_mask]
    
    print(f"Loaded {len(output['data'])} samples from {data_path}")
    
    return output


def predict(model, data, device='cpu', batch_size=128):
    """
    Run inference on data.
    
    Args:
        model: DeepBeat model
        data: Input data (N, 800, 1)
        device: Device to run on
        batch_size: Batch size for inference
    
    Returns:
        Dictionary with predictions
    """
    model.eval()
    
    qa_predictions = []
    rhythm_predictions = []
    qa_probabilities = []
    rhythm_probabilities = []
    
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_data = torch.FloatTensor(data[start_idx:end_idx]).to(device)
            
            outputs = model(batch_data)
            
            # Get predictions and probabilities
            qa_prob = outputs['qa_output'].cpu().numpy()
            rhythm_prob = outputs['rhythm_output'].cpu().numpy()
            
            qa_pred = np.argmax(qa_prob, axis=1)
            rhythm_pred = np.argmax(rhythm_prob, axis=1)
            
            qa_predictions.extend(qa_pred)
            rhythm_predictions.extend(rhythm_pred)
            qa_probabilities.extend(qa_prob)
            rhythm_probabilities.extend(rhythm_prob)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {end_idx}/{num_samples} samples...")
    
    return {
        'qa_predictions': np.array(qa_predictions),
        'rhythm_predictions': np.array(rhythm_predictions),
        'qa_probabilities': np.array(qa_probabilities),
        'rhythm_probabilities': np.array(rhythm_probabilities)
    }


def evaluate(predictions, ground_truth):
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions: Dictionary with predictions
        ground_truth: Dictionary with true labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    if ground_truth['qa_label'] is not None:
        qa_true = np.argmax(ground_truth['qa_label'], axis=1)
        qa_pred = predictions['qa_predictions']
        qa_accuracy = (qa_true == qa_pred).mean()
        metrics['qa_accuracy'] = qa_accuracy
        
        print(f"\nQA Classification Results:")
        print(f"  Accuracy: {qa_accuracy:.4f}")
        
        # Per-class accuracy
        for cls in range(3):
            mask = qa_true == cls
            if mask.sum() > 0:
                cls_acc = (qa_pred[mask] == cls).mean()
                print(f"  Class {cls} accuracy: {cls_acc:.4f} ({mask.sum()} samples)")
    
    if ground_truth['rhythm'] is not None:
        rhythm_true = np.argmax(ground_truth['rhythm'], axis=1)
        rhythm_pred = predictions['rhythm_predictions']
        rhythm_accuracy = (rhythm_true == rhythm_pred).mean()
        metrics['rhythm_accuracy'] = rhythm_accuracy
        
        print(f"\nRhythm Classification Results:")
        print(f"  Accuracy: {rhythm_accuracy:.4f}")
        
        # Per-class accuracy
        for cls in range(2):
            mask = rhythm_true == cls
            if mask.sum() > 0:
                cls_acc = (rhythm_pred[mask] == cls).mean()
                print(f"  Class {cls} accuracy: {cls_acc:.4f} ({mask.sum()} samples)")
    
    return metrics


def save_predictions(predictions, output_path):
    """Save predictions to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)
    
    print(f"\nPredictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='DeepBeat Model Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file (.npz)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save predictions (optional)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Load model
    print("=" * 60)
    print("Loading model...")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model_path, device)
    
    # Load data
    print("\nLoading data...")
    data_dict = load_data(args.data_path)
    
    # Run inference
    print("\nRunning inference...")
    print("=" * 60)
    predictions = predict(model, data_dict['data'], device, args.batch_size)
    print(f"Inference complete!")
    
    # Evaluate if ground truth is available
    if data_dict['qa_label'] is not None or data_dict['rhythm'] is not None:
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print("=" * 60)
        metrics = evaluate(predictions, data_dict)
    
    # Save predictions if output path provided
    if args.output_path:
        save_predictions(predictions, args.output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Total samples processed: {len(predictions['qa_predictions'])}")
    print(f"\nQA Predictions distribution:")
    for cls in range(3):
        count = (predictions['qa_predictions'] == cls).sum()
        print(f"  Class {cls}: {count} ({count/len(predictions['qa_predictions'])*100:.1f}%)")
    print(f"\nRhythm Predictions distribution:")
    for cls in range(2):
        count = (predictions['rhythm_predictions'] == cls).sum()
        print(f"  Class {cls}: {count} ({count/len(predictions['rhythm_predictions'])*100:.1f}%)")


if __name__ == "__main__":
    main()
