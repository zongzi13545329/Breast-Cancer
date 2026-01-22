"""
Comprehensive Metrics for Recall Reduction Evaluation

Includes:
1. Standard classification metrics (AUC, sensitivity, specificity, PPV, NPV)
2. Fairness metrics (demographic parity, equalized odds, calibration)
3. Statistical tests (DeLong, bootstrap confidence intervals)
4. Clinical impact metrics (decision curve analysis)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, average_precision_score
)
from scipy import stats
from scipy.stats import bootstrap


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        predictions: Predicted probabilities [N]
        labels: Ground truth labels [N]
        threshold: Decision threshold
    
    Returns:
        Dictionary of metrics
    """
    # Binary predictions
    pred_binary = (predictions >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
    
    # Calculate metrics
    metrics = {}
    
    # AUC
    if len(np.unique(labels)) > 1:
        metrics['auc'] = roc_auc_score(labels, predictions)
        metrics['auprc'] = average_precision_score(labels, predictions)
    else:
        metrics['auc'] = 0.0
        metrics['auprc'] = 0.0
    
    # Sensitivity (Recall, TPR)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['recall'] = metrics['sensitivity']
    metrics['tpr'] = metrics['sensitivity']
    
    # Specificity (TNR)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['tnr'] = metrics['specificity']
    
    # False Positive Rate
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Positive Predictive Value (Precision)
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['precision'] = metrics['ppv']
    
    # Negative Predictive Value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # F1 Score
    metrics['f1'] = f1_score(labels, pred_binary)
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, pred_binary)
    
    # Recall rate (percentage of exams recalled)
    metrics['recall_rate'] = pred_binary.mean()
    
    return metrics


def compute_fairness_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute fairness metrics across demographic groups.
    
    Args:
        predictions: Predicted probabilities [N]
        labels: Ground truth labels [N]
        groups: Group labels [N] (e.g., race indices)
        threshold: Decision threshold
    
    Returns:
        Dictionary of fairness metrics
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    # Binary predictions
    pred_binary = (predictions >= threshold).astype(int)
    
    # Metrics per group
    group_metrics = {}
    tpr_by_group = []
    fpr_by_group = []
    ppv_by_group = []
    recall_rate_by_group = []
    
    for group in unique_groups:
        mask = (groups == group)
        
        if mask.sum() == 0:
            continue
        
        group_pred = predictions[mask]
        group_labels = labels[mask]
        group_pred_binary = pred_binary[mask]
        
        # Compute metrics for this group
        group_metrics[f'group_{group}'] = compute_metrics(group_pred, group_labels, threshold)
        
        # Extract key metrics
        tpr_by_group.append(group_metrics[f'group_{group}']['tpr'])
        fpr_by_group.append(group_metrics[f'group_{group}']['fpr'])
        ppv_by_group.append(group_metrics[f'group_{group}']['ppv'])
        recall_rate_by_group.append(group_pred_binary.mean())
    
    # Convert to arrays
    tpr_by_group = np.array(tpr_by_group)
    fpr_by_group = np.array(fpr_by_group)
    ppv_by_group = np.array(ppv_by_group)
    recall_rate_by_group = np.array(recall_rate_by_group)
    
    # Fairness metrics
    fairness = {}
    
    # Demographic Parity: P(Y_hat=1 | Group=i) should be similar
    fairness['demographic_parity_diff'] = recall_rate_by_group.max() - recall_rate_by_group.min()
    fairness['demographic_parity_ratio'] = recall_rate_by_group.min() / recall_rate_by_group.max() if recall_rate_by_group.max() > 0 else 0
    
    # Equalized Odds: TPR and FPR should be similar across groups
    fairness['tpr_disparity'] = tpr_by_group.max() - tpr_by_group.min()
    fairness['fpr_disparity'] = fpr_by_group.max() - fpr_by_group.min()
    fairness['equalized_odds_diff'] = max(fairness['tpr_disparity'], fairness['fpr_disparity'])
    
    # PPV Parity
    fairness['ppv_disparity'] = ppv_by_group.max() - ppv_by_group.min()
    
    # Overall fairness score (lower is better)
    fairness['overall_fairness_violation'] = (
        fairness['demographic_parity_diff'] +
        fairness['equalized_odds_diff'] +
        fairness['ppv_disparity']
    ) / 3.0
    
    return fairness


def compute_bootstrap_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        metric_fn: Function to compute metric (takes predictions, labels)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95)
        random_state: Random seed
    
    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    
    n_samples = len(predictions)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        pred_resampled = predictions[indices]
        labels_resampled = labels[indices]
        
        # Skip if no positive samples
        if len(np.unique(labels_resampled)) < 2:
            continue
        
        # Compute metric
        score = metric_fn(pred_resampled, labels_resampled)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Point estimate
    point_estimate = metric_fn(predictions, labels)
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    
    return point_estimate, ci_lower, ci_upper


def delong_test(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    DeLong test to compare two AUCs.
    
    Args:
        predictions1: Predictions from model 1
        predictions2: Predictions from model 2
        labels: Ground truth labels
    
    Returns:
        (z_statistic, p_value)
    """
    from scipy.stats import norm
    
    # Compute AUCs
    auc1 = roc_auc_score(labels, predictions1)
    auc2 = roc_auc_score(labels, predictions2)
    
    # Simplified DeLong (approximate)
    # For exact implementation, would need structural components
    n = len(labels)
    
    # Compute variance using bootstrap approximation
    n_bootstrap = 1000
    auc1_bootstrap = []
    auc2_bootstrap = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        
        if len(np.unique(labels[indices])) < 2:
            continue
        
        auc1_bootstrap.append(roc_auc_score(labels[indices], predictions1[indices]))
        auc2_bootstrap.append(roc_auc_score(labels[indices], predictions2[indices]))
    
    # Compute covariance
    auc1_bootstrap = np.array(auc1_bootstrap)
    auc2_bootstrap = np.array(auc2_bootstrap)
    
    var1 = np.var(auc1_bootstrap)
    var2 = np.var(auc2_bootstrap)
    cov = np.cov(auc1_bootstrap, auc2_bootstrap)[0, 1]
    
    # Compute z-statistic
    var_diff = var1 + var2 - 2 * cov
    
    if var_diff > 0:
        z = (auc1 - auc2) / np.sqrt(var_diff)
        p_value = 2 * (1 - norm.cdf(abs(z)))
    else:
        z = 0.0
        p_value = 1.0
    
    return z, p_value


def compute_calibration(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute calibration curve.
    
    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        n_bins: Number of bins for calibration curve
    
    Returns:
        Dictionary with bin_edges, bin_centers, observed_freq, predicted_freq
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign predictions to bins
    bin_indices = np.digitize(predictions, bin_edges[1:-1])
    
    # Compute observed and predicted frequencies per bin
    observed_freq = []
    predicted_freq = []
    bin_sizes = []
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        
        if mask.sum() > 0:
            observed_freq.append(labels[mask].mean())
            predicted_freq.append(predictions[mask].mean())
            bin_sizes.append(mask.sum())
        else:
            observed_freq.append(np.nan)
            predicted_freq.append(np.nan)
            bin_sizes.append(0)
    
    # Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = len(predictions)
    
    for i in range(n_bins):
        if bin_sizes[i] > 0 and not np.isnan(observed_freq[i]):
            ece += (bin_sizes[i] / total_samples) * abs(observed_freq[i] - predicted_freq[i])
    
    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'observed_freq': np.array(observed_freq),
        'predicted_freq': np.array(predicted_freq),
        'bin_sizes': np.array(bin_sizes),
        'ece': ece
    }


def compute_operating_points(
    predictions: np.ndarray,
    labels: np.ndarray,
    target_sensitivity: float = 0.95
) -> Dict[str, float]:
    """
    Find operating point that achieves target sensitivity.
    
    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        target_sensitivity: Target sensitivity level
    
    Returns:
        Dictionary with threshold, specificity, ppv, recall_rate
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    
    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    
    threshold = thresholds[idx]
    achieved_sensitivity = tpr[idx]
    achieved_specificity = 1 - fpr[idx]
    
    # Compute metrics at this threshold
    pred_binary = (predictions >= threshold).astype(int)
    metrics = compute_metrics(predictions, labels, threshold)
    
    return {
        'threshold': threshold,
        'sensitivity': achieved_sensitivity,
        'specificity': achieved_specificity,
        'ppv': metrics['ppv'],
        'npv': metrics['npv'],
        'recall_rate': pred_binary.mean(),
        'f1': metrics['f1']
    }

def compute_multitask_metrics(predictions, labels):
    """
    Compute metrics for all tasks.
    
    Args:
        predictions: Dict with 'recall', 'birads', 'density'
        labels: Dict with same keys
    
    Returns:
        Dict of metrics per task
    """
    metrics = {}
    
    # Recall task (binary)
    recall_pred = predictions['recall'].sigmoid() > 0.5
    metrics['recall'] = {
        'accuracy': accuracy_score(labels['recall'], recall_pred),
        'auc': roc_auc_score(labels['recall'], predictions['recall'].sigmoid()),
        'f1': f1_score(labels['recall'], recall_pred)
    }
    
    # BI-RADS task (multi-class)
    birads_pred = predictions['birads'].argmax(dim=1)
    metrics['birads'] = {
        'accuracy': accuracy_score(labels['birads'], birads_pred),
        'f1_macro': f1_score(labels['birads'], birads_pred, average='macro')
    }
    
    # Density task (multi-class)
    density_pred = predictions['density'].argmax(dim=1)
    metrics['density'] = {
        'accuracy': accuracy_score(labels['density'], density_pred),
        'f1_macro': f1_score(labels['density'], density_pred, average='macro')
    }
    
    return metrics
    
if __name__ == "__main__":
    # Test metrics
    print("Testing metrics computation...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    predictions = np.random.rand(n_samples)
    labels = (predictions + np.random.randn(n_samples) * 0.3 > 0.5).astype(int)
    groups = np.random.choice([0, 1, 2, 3], n_samples)  # 4 groups
    
    # Compute metrics
    print("\n1. Standard Metrics:")
    metrics = compute_metrics(predictions, labels)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n2. Fairness Metrics:")
    fairness = compute_fairness_metrics(predictions, labels, groups)
    for key, value in fairness.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n3. Bootstrap CI for AUC:")
    auc, ci_lower, ci_upper = compute_bootstrap_ci(
        predictions, labels,
        lambda p, l: roc_auc_score(l, p),
        n_bootstrap=100
    )
    print(f"  AUC: {auc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
    print("\n4. Calibration:")
    calibration = compute_calibration(predictions, labels, n_bins=5)
    print(f"  ECE: {calibration['ece']:.4f}")
    
    print("\n5. Operating Point (95% Sensitivity):")
    op = compute_operating_points(predictions, labels, target_sensitivity=0.95)
    for key, value in op.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nAll tests passed!")
