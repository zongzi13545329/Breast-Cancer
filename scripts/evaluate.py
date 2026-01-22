"""
Comprehensive Evaluation Script

Evaluates trained models with:
1. Standard performance metrics
2. Fairness analysis by race
3. Statistical significance tests
4. Calibration analysis
5. Visualization of results
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import EMBEDDataset
from models.temporal_model import create_temporal_model
from models.fairness_model import create_fairness_model
from utils.metrics import (
    compute_metrics, compute_fairness_metrics, compute_bootstrap_ci,
    compute_calibration, compute_operating_points, delong_test
)
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint


class Evaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, config: Dict, checkpoint_path: str, output_dir: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(self.output_dir / 'evaluation.log')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test dataset
        self.test_dataset = self._create_dataset()
        self.test_loader = self._create_dataloader()
        
        # Create and load model
        self.model = self._create_and_load_model()
        
    def _create_dataset(self):
        """Create test dataset."""
        data_config = self.config['data']
        
        dataset = EMBEDDataset(
            data_root=data_config['root_dir'],
            metadata_csv=os.path.join(data_config['root_dir'], 'metadata.csv'),
            split='test',
            image_size=data_config['image_size'],
            use_temporal=self.config['model']['use_temporal'],
            max_temporal_gap_years=data_config['temporal_window_years'],
            augment=False,
            normalize=True
        )
        
        self.logger.info(f"Test dataset: {len(dataset)} samples")
        return dataset
    
    def _create_dataloader(self):
        """Create test dataloader."""
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        return loader
    
    def _create_and_load_model(self):
        """Create model and load checkpoint."""
        model_config = self.config['model']
        
        if model_config['use_fairness']:
            fairness_method = self.config['fairness'].get('constraint', 'race_conditional')
            model = create_fairness_model(model_config, method=fairness_method)
        else:
            model = create_temporal_model(model_config)
        
        model = model.to(self.device)
        
        # Load checkpoint
        load_checkpoint(self.checkpoint_path, model, device=str(self.device))
        
        model.eval()
        return model
    
    @torch.no_grad()
    def collect_predictions(self):
        """Collect predictions on test set."""
        all_predictions = []
        all_labels = []
        all_races = []
        all_exam_ids = []
        all_has_prior = []
        
        self.logger.info("Collecting predictions...")
        
        for batch in tqdm(self.test_loader, desc="Inference"):
            current_img = batch['current_img'].to(self.device)
            prior_img = batch['prior_img'].to(self.device) if self.config['model']['use_temporal'] else None
            labels = batch['label']
            race_labels = batch['race']
            exam_ids = batch['exam_id']
            has_prior = batch['has_prior']
            
            # Forward pass
            if self.config['model']['use_fairness']:
                output = self.model(current_img, prior_img, race_labels.to(self.device))
            else:
                output = self.model(current_img, prior_img)
            
            logits = output['logits'].squeeze()
            predictions = torch.sigmoid(logits)
            
            # Collect
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())
            all_races.append(race_labels.numpy())
            all_exam_ids.extend(exam_ids)
            all_has_prior.append(has_prior.numpy())
        
        # Concatenate
        self.predictions = np.concatenate(all_predictions)
        self.labels = np.concatenate(all_labels)
        self.races = np.concatenate(all_races)
        self.exam_ids = all_exam_ids
        self.has_prior = np.concatenate(all_has_prior)
        
        self.logger.info(f"Collected {len(self.predictions)} predictions")
    
    def compute_overall_metrics(self):
        """Compute overall performance metrics."""
        self.logger.info("\n" + "="*60)
        self.logger.info("OVERALL PERFORMANCE METRICS")
        self.logger.info("="*60)
        
        # Standard metrics at default threshold
        metrics = compute_metrics(self.predictions, self.labels, threshold=0.5)
        
        for key, value in metrics.items():
            self.logger.info(f"{key:20s}: {value:.4f}")
        
        # Bootstrap confidence intervals for AUC
        auc, ci_lower, ci_upper = compute_bootstrap_ci(
            self.predictions, self.labels,
            lambda p, l: compute_metrics(p, l)['auc'],
            n_bootstrap=1000
        )
        self.logger.info(f"\nAUC with 95% CI: {auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Operating point at 95% sensitivity
        op = compute_operating_points(self.predictions, self.labels, target_sensitivity=0.95)
        self.logger.info(f"\nOperating Point (95% Sensitivity):")
        for key, value in op.items():
            self.logger.info(f"  {key:20s}: {value:.4f}")
        
        # Calibration
        calibration = compute_calibration(self.predictions, self.labels, n_bins=10)
        self.logger.info(f"\nExpected Calibration Error: {calibration['ece']:.4f}")
        
        # Save overall metrics
        with open(self.output_dir / 'overall_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def compute_fairness_analysis(self):
        """Compute fairness metrics by race."""
        self.logger.info("\n" + "="*60)
        self.logger.info("FAIRNESS ANALYSIS BY RACE")
        self.logger.info("="*60)
        
        # Overall fairness metrics
        fairness = compute_fairness_metrics(self.predictions, self.labels, self.races)
        
        self.logger.info("\nOverall Fairness Metrics:")
        for key, value in fairness.items():
            if not key.startswith('group_'):
                self.logger.info(f"{key:30s}: {value:.4f}")
        
        # Per-race metrics
        race_names = ['African American', 'White', 'Asian', 'Hispanic', 'Other']
        
        self.logger.info("\nPer-Race Performance:")
        race_results = []
        
        for race_idx in np.unique(self.races):
            mask = (self.races == race_idx)
            
            if mask.sum() == 0:
                continue
            
            race_pred = self.predictions[mask]
            race_labels = self.labels[mask]
            
            metrics = compute_metrics(race_pred, race_labels)
            
            race_name = race_names[race_idx] if race_idx < len(race_names) else f'Group {race_idx}'
            
            self.logger.info(f"\n{race_name} (n={mask.sum()}):")
            self.logger.info(f"  AUC:         {metrics['auc']:.4f}")
            self.logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            self.logger.info(f"  Specificity: {metrics['specificity']:.4f}")
            self.logger.info(f"  PPV:         {metrics['ppv']:.4f}")
            self.logger.info(f"  FPR:         {metrics['fpr']:.4f}")
            self.logger.info(f"  Recall Rate: {metrics['recall_rate']:.4f}")
            
            race_results.append({
                'race': race_name,
                'n': mask.sum(),
                **metrics
            })
        
        # Save to DataFrame
        race_df = pd.DataFrame(race_results)
        race_df.to_csv(self.output_dir / 'race_stratified_metrics.csv', index=False)
        
        # Statistical test for disparity
        self.logger.info("\nStatistical Significance Tests:")
        
        # Compare AUCs between racial groups
        race_pairs = [(0, 1), (0, 2), (0, 3)]  # AA vs White, AA vs Asian, AA vs Hispanic
        
        for race1, race2 in race_pairs:
            if race1 >= len(race_names) or race2 >= len(race_names):
                continue
            
            mask1 = (self.races == race1)
            mask2 = (self.races == race2)
            
            if mask1.sum() == 0 or mask2.sum() == 0:
                continue
            
            # Combine for comparison
            combined_mask = mask1 | mask2
            combined_pred = self.predictions[combined_mask]
            combined_labels = self.labels[combined_mask]
            combined_races = self.races[combined_mask]
            
            # Create binary race indicator
            race_indicator = (combined_races == race1).astype(float)
            
            # Simple t-test on predictions by race (approximate)
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(
                combined_pred[combined_races == race1],
                combined_pred[combined_races == race2]
            )
            
            self.logger.info(f"  {race_names[race1]} vs {race_names[race2]}: p={p_value:.4f}")
        
        return fairness
    
    def analyze_temporal_impact(self):
        """Analyze impact of temporal information."""
        if not self.config['model']['use_temporal']:
            self.logger.info("\nSkipping temporal analysis (model doesn't use temporal info)")
            return
        
        self.logger.info("\n" + "="*60)
        self.logger.info("TEMPORAL INFORMATION IMPACT")
        self.logger.info("="*60)
        
        # Split by whether prior was available
        with_prior_mask = self.has_prior
        without_prior_mask = ~self.has_prior
        
        self.logger.info(f"\nSamples with prior: {with_prior_mask.sum()}")
        self.logger.info(f"Samples without prior: {without_prior_mask.sum()}")
        
        # Metrics with prior
        if with_prior_mask.sum() > 0:
            metrics_with = compute_metrics(
                self.predictions[with_prior_mask],
                self.labels[with_prior_mask]
            )
            self.logger.info(f"\nPerformance WITH prior:")
            self.logger.info(f"  AUC: {metrics_with['auc']:.4f}")
            self.logger.info(f"  Specificity: {metrics_with['specificity']:.4f}")
        
        # Metrics without prior
        if without_prior_mask.sum() > 0:
            metrics_without = compute_metrics(
                self.predictions[without_prior_mask],
                self.labels[without_prior_mask]
            )
            self.logger.info(f"\nPerformance WITHOUT prior:")
            self.logger.info(f"  AUC: {metrics_without['auc']:.4f}")
            self.logger.info(f"  Specificity: {metrics_without['specificity']:.4f}")
        
        # Improvement
        if with_prior_mask.sum() > 0 and without_prior_mask.sum() > 0:
            auc_improvement = metrics_with['auc'] - metrics_without['auc']
            spec_improvement = metrics_with['specificity'] - metrics_without['specificity']
            
            self.logger.info(f"\nImprovement from temporal information:")
            self.logger.info(f"  AUC: +{auc_improvement:.4f}")
            self.logger.info(f"  Specificity: +{spec_improvement:.4f}")
    
    def plot_roc_curves(self):
        """Plot ROC curves overall and by race."""
        from sklearn.metrics import roc_curve
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Overall ROC
        fpr, tpr, _ = roc_curve(self.labels, self.predictions)
        auc = compute_metrics(self.predictions, self.labels)['auc']
        
        ax1.plot(fpr, tpr, label=f'Overall (AUC={auc:.3f})', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve - Overall', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        
        # By race
        race_names = ['African American', 'White', 'Asian', 'Hispanic', 'Other']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for race_idx in np.unique(self.races):
            mask = (self.races == race_idx)
            
            if mask.sum() < 10:  # Skip if too few samples
                continue
            
            fpr_r, tpr_r, _ = roc_curve(self.labels[mask], self.predictions[mask])
            auc_r = compute_metrics(self.predictions[mask], self.labels[mask])['auc']
            
            race_name = race_names[race_idx] if race_idx < len(race_names) else f'Group {race_idx}'
            color = colors[race_idx] if race_idx < len(colors) else 'gray'
            
            ax2.plot(fpr_r, tpr_r, label=f'{race_name} (AUC={auc_r:.3f})',
                    linewidth=2, color=color)
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('ROC Curves by Race', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"\nROC curves saved to {self.output_dir / 'roc_curves.png'}")
    
    def plot_calibration(self):
        """Plot calibration curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Overall calibration
        calib = compute_calibration(self.predictions, self.labels, n_bins=10)
        
        valid_bins = ~np.isnan(calib['observed_freq'])
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
        ax1.scatter(calib['predicted_freq'][valid_bins],
                   calib['observed_freq'][valid_bins],
                   s=calib['bin_sizes'][valid_bins],
                   alpha=0.6,
                   label=f'ECE={calib["ece"]:.4f}')
        ax1.set_xlabel('Predicted Probability', fontsize=12)
        ax1.set_ylabel('Observed Frequency', fontsize=12)
        ax1.set_title('Calibration Curve - Overall', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # By race
        race_names = ['African American', 'White', 'Asian', 'Hispanic']
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for race_idx in range(min(4, len(np.unique(self.races)))):
            mask = (self.races == race_idx)
            
            if mask.sum() < 50:
                continue
            
            calib_r = compute_calibration(self.predictions[mask], self.labels[mask], n_bins=10)
            valid_bins_r = ~np.isnan(calib_r['observed_freq'])
            
            race_name = race_names[race_idx]
            color = colors[race_idx]
            
            ax2.plot(calib_r['predicted_freq'][valid_bins_r],
                    calib_r['observed_freq'][valid_bins_r],
                    marker='o', label=f'{race_name} (ECE={calib_r["ece"]:.4f})',
                    color=color, linewidth=2, markersize=6)
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Observed Frequency', fontsize=12)
        ax2.set_title('Calibration by Race', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Calibration curves saved to {self.output_dir / 'calibration.png'}")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        # Collect predictions
        self.collect_predictions()
        
        # Compute metrics
        self.compute_overall_metrics()
        
        # Fairness analysis
        self.compute_fairness_analysis()
        
        # Temporal impact
        self.analyze_temporal_impact()
        
        # Plots
        self.plot_roc_curves()
        self.plot_calibration()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate EMBED recall reduction model")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint, args.output_dir)
    
    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
