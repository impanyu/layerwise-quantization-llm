#!/usr/bin/env python3
"""
CUDA Environment Diagnostic and Fix Script

This script helps diagnose CUDA environment issues and provides fixes.
"""

import os
import sys
import subprocess
import torch

def check_nvidia_driver():
    """Check if NVIDIA driver is working."""
    print("🔍 Checking NVIDIA driver...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA driver is working")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    print(f"📋 {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi failed:")
            print(result.stderr)
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"❌ nvidia-smi not found or timed out: {e}")
        return False

def check_cuda_environment():
    """Check CUDA environment variables."""
    print("\n🔍 Checking CUDA environment variables...")
    
    cuda_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER', 
        'CUDA_HOME',
        'CUDA_PATH',
        'LD_LIBRARY_PATH'
    ]
    
    issues = []
    for var in cuda_vars:
        value = os.environ.get(var)
        if value is not None:
            print(f"📋 {var}={value}")
            # Check for potential issues
            if var == 'CUDA_VISIBLE_DEVICES' and value == '':
                issues.append(f"❌ {var} is set to empty string (this hides all GPUs)")
        else:
            print(f"📋 {var}=<not set>")
    
    return issues

def check_pytorch_cuda():
    """Check PyTorch CUDA functionality."""
    print("\n🔍 Checking PyTorch CUDA...")
    
    print(f"📋 PyTorch version: {torch.__version__}")
    print(f"📋 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"📋 CUDA version: {torch.version.cuda}")
        print(f"📋 Device count: {torch.cuda.device_count()}")
        
        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"📋 GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
        except Exception as e:
            print(f"❌ Error getting GPU properties: {e}")
            return False
        
        # Test basic CUDA operations
        try:
            print("🧪 Testing basic CUDA operations...")
            torch.cuda.init()
            device = torch.cuda.current_device()
            print(f"📋 Current device: {device}")
            
            # Test tensor operations
            x = torch.tensor([1.0, 2.0]).cuda()
            y = x + 1
            result = y.cpu().numpy()
            print(f"📋 Test operation result: {result}")
            print("✅ Basic CUDA operations work")
            return True
            
        except Exception as e:
            print(f"❌ CUDA operations failed: {e}")
            return False
    else:
        print("❌ CUDA is not available to PyTorch")
        return False

def suggest_fixes(nvidia_ok, cuda_env_issues, pytorch_cuda_ok):
    """Suggest fixes based on diagnostic results."""
    print("\n🛠️  Suggested Fixes:")
    
    if not nvidia_ok:
        print("1. Install or update NVIDIA drivers")
        print("2. Restart your system after driver installation")
        return
    
    if cuda_env_issues:
        print("1. Fix CUDA environment variables:")
        for issue in cuda_env_issues:
            print(f"   {issue}")
            if "CUDA_VISIBLE_DEVICES" in issue:
                print("   Fix: unset CUDA_VISIBLE_DEVICES OR set it to '0' or '0,1' etc.")
        
    if not pytorch_cuda_ok:
        print("2. Try these environment fixes:")
        print("   export CUDA_VISIBLE_DEVICES=0  # or your GPU IDs")
        print("   unset CUDA_VISIBLE_DEVICES  # if you want all GPUs")
        print("   # Then restart your Python session")
        
    print("\n3. Alternative: Run training with explicit device:")
    print("   python ./layerwise/train_layerwise_router.py ... --device cuda")
    
    print("\n4. If issues persist, restart Python session and try:")
    print("   import torch")
    print("   torch.cuda.init()  # Force CUDA initialization")
    print("   torch.cuda.current_device()")

def main():
    print("CUDA Environment Diagnostic Tool")
    print("=" * 50)
    
    # Run diagnostics
    nvidia_ok = check_nvidia_driver()
    cuda_env_issues = check_cuda_environment() 
    pytorch_cuda_ok = check_pytorch_cuda()
    
    # Provide suggestions
    suggest_fixes(nvidia_ok, cuda_env_issues, pytorch_cuda_ok)
    
    # Summary
    print("\n📊 Summary:")
    print(f"   NVIDIA Driver: {'✅' if nvidia_ok else '❌'}")
    print(f"   CUDA Environment: {'✅' if not cuda_env_issues else '❌'}")
    print(f"   PyTorch CUDA: {'✅' if pytorch_cuda_ok else '❌'}")
    
    if nvidia_ok and not cuda_env_issues and pytorch_cuda_ok:
        print("\n🎉 CUDA environment looks good! The issue might be elsewhere.")
    else:
        print("\n⚠️  Please fix the issues above before running training.")

if __name__ == "__main__":
    main()
