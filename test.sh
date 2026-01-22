#!/bin/bash
# EMBED Project - Quick Test Script
# 这个脚本会自动运行一系列测试来验证代码是否正确

set -e  # 遇到错误立即退出

echo "=================================================="
echo "EMBED Project - Automated Testing Script"
echo "=================================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试函数
test_passed() {
    echo -e "${GREEN}✅ PASSED${NC}: $1"
}

test_failed() {
    echo -e "${RED}❌ FAILED${NC}: $1"
    exit 1
}

test_warning() {
    echo -e "${YELLOW}⚠️  WARNING${NC}: $1"
}

# ============================================================================
# Test 1: Python Environment
# ============================================================================
echo "Test 1: Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "  Found: $PYTHON_VERSION"
    test_passed "Python is installed"
else
    test_failed "Python is not installed"
fi
echo ""

# ============================================================================
# Test 2: Required Packages
# ============================================================================
echo "Test 2: Checking required packages..."

packages=("torch" "torchvision" "timm" "pandas" "PIL" "pydicom" "sklearn" "yaml" "tqdm")
missing_packages=()

for pkg in "${packages[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        echo "  ✓ $pkg"
    else
        echo "  ✗ $pkg (MISSING)"
        missing_packages+=("$pkg")
    fi
done

if [ ${#missing_packages[@]} -eq 0 ]; then
    test_passed "All required packages are installed"
else
    test_warning "Missing packages: ${missing_packages[*]}"
    echo "  Run: pip install torch torchvision timm pandas pillow pydicom scikit-learn pyyaml tqdm"
fi
echo ""

# ============================================================================
# Test 3: File Structure
# ============================================================================
echo "Test 3: Checking file structure..."

required_files=(
    "models/temporal_model.py"
    "models/fairness_model.py"
    "models/__init__.py"
    "data/dataset.py"
    "data/__init__.py"
    "utils/metrics.py"
    "utils/logger.py"
    "utils/__init__.py"
    "scripts/train.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    test_passed "All required files exist"
else
    test_failed "Missing files: ${missing_files[*]}"
fi
echo ""

# ============================================================================
# Test 4: Python Syntax Check
# ============================================================================
echo "Test 4: Checking Python syntax..."

syntax_errors=()
for file in "${required_files[@]}"; do
    if [[ $file == *.py ]]; then
        if python -m py_compile "$file" 2>/dev/null; then
            echo "  ✓ $file"
        else
            echo "  ✗ $file (SYNTAX ERROR)"
            syntax_errors+=("$file")
        fi
    fi
done

if [ ${#syntax_errors[@]} -eq 0 ]; then
    test_passed "No syntax errors found"
else
    test_failed "Syntax errors in: ${syntax_errors[*]}"
fi
echo ""

# ============================================================================
# Test 5: Import Test
# ============================================================================
echo "Test 5: Testing module imports..."

python << 'EOF'
import sys
import traceback

def test_import(module_path, names):
    try:
        module = __import__(module_path, fromlist=names)
        for name in names:
            getattr(module, name)
        print(f"  ✓ {module_path}: {', '.join(names)}")
        return True
    except Exception as e:
        print(f"  ✗ {module_path}: {e}")
        traceback.print_exc()
        return False

all_passed = True

# Test temporal model
all_passed &= test_import('models.temporal_model', 
    ['TemporalSiameseNetwork', 'create_temporal_model'])

# Test fairness model
all_passed &= test_import('models.fairness_model',
    ['FairnessTemporalModel', 'MultiTaskLoss', 'create_fairness_model'])

# Test dataset
all_passed &= test_import('data.dataset',
    ['EMBEDLongitudinalDataset'])

# Test utils
all_passed &= test_import('utils.metrics',
    ['compute_metrics', 'compute_fairness_metrics'])

all_passed &= test_import('utils.logger',
    ['setup_logger', 'save_checkpoint', 'load_checkpoint'])

sys.exit(0 if all_passed else 1)
EOF

if [ $? -eq 0 ]; then
    test_passed "All module imports successful"
else
    test_failed "Import errors detected - check code fixes"
fi
echo ""

# ============================================================================
# Test 6: Model Instantiation
# ============================================================================
echo "Test 6: Testing model instantiation..."

python << 'EOF'
import torch
from models.fairness_model import FairnessTemporalModel, MultiTaskLoss

try:
    # Test model creation
    model = FairnessTemporalModel(
        backbone='resnet50',
        num_classes=1,
        num_birads=5,
        num_density=4,
        num_races=4,
        dropout=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with small inputs
    current = torch.randn(2, 1, 256, 128)
    prior = torch.randn(2, 1, 256, 128)
    race_labels = torch.tensor([0, 1])
    
    with torch.no_grad():
        output = model(current, prior, race_labels)
    
    print(f"  ✓ Forward pass successful")
    print(f"  Output keys: {list(output.keys())}")
    print(f"  Recall shape: {output['recall'].shape}")
    print(f"  BI-RADS shape: {output['birads'].shape}")
    print(f"  Density shape: {output['density'].shape}")
    
    # Test loss function
    loss_fn = MultiTaskLoss()
    labels = {
        'recall': torch.randint(0, 2, (2, 1)).float(),
        'birads': torch.randint(0, 5, (2,)),
        'density': torch.randint(0, 4, (2,))
    }
    losses = loss_fn(output, labels, race_labels)
    print(f"  ✓ Loss computation successful")
    print(f"  Total loss: {losses['total'].item():.4f}")
    
    print("\n✅ Model instantiation and forward pass: SUCCESS")
    
except Exception as e:
    print(f"\n❌ Model instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

if [ $? -eq 0 ]; then
    test_passed "Model instantiation and forward pass"
else
    test_failed "Model instantiation failed"
fi
echo ""

# ============================================================================
# Test 7: Configuration Files
# ============================================================================
echo "Test 7: Checking configuration files..."

config_files=(
    "configs/base_config.yaml"
    "configs/baseline.yaml"
    "configs/temporal_model.yaml"
    "configs/fairness_model.yaml"
)

missing_configs=()
for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        missing_configs+=("$file")
    fi
done

if [ ${#missing_configs[@]} -eq 0 ]; then
    test_passed "All configuration files exist"
else
    test_warning "Missing configs: ${missing_configs[*]}"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=================================================="
echo "Test Summary"
echo "=================================================="
echo ""
echo "✅ All critical tests passed!"
echo ""
echo "Next steps:"
echo "1. Set your data paths:"
echo "   export CLINICAL_CSV='/path/to/clinical.csv'"
echo "   export METADATA_CSV='/path/to/metadata.csv'"
echo "   export IMAGE_ROOT='/path/to/images/'"
echo ""
echo "2. Run a quick test:"
echo "   python train.py \\"
echo "     --clinical_csv \$CLINICAL_CSV \\"
echo "     --metadata_csv \$METADATA_CSV \\"
echo "     --image_root \$IMAGE_ROOT \\"
echo "     --batch_size 2 \\"
echo "     --num_epochs 1 \\"
echo "     --image_size 128 64 \\"
echo "     --no-temporal-attention \\"
echo "     --no-fairness-adapter"
echo ""
echo "3. For full training, see RUN_COMMANDS.md"
echo ""