import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

def inspect_model_devices(model, prefix="", max_depth=3, current_depth=0):
    """
    Recursively inspect all properties of a PyTorch Lightning model to find
    which tensors are on CPU vs CUDA devices and check density/sparsity.
    
    Args:
        model: The model/object to inspect
        prefix: String prefix for nested attributes
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
    
    Returns:
        Dict with categorized results
    """
    results = {
        'cuda_tensors': [],
        'cpu_tensors': [],
        'mixed_tensors': [],  # For modules with tensors on different devices
        'sparse_tensors': [],  # Sparse tensors
        'non_contiguous_tensors': [],  # Non-contiguous (non-dense) tensors
        'problematic_tensors': [],  # CPU or sparse or non-contiguous
        'non_tensor_attrs': [],
        'errors': []
    }
    
    if current_depth >= max_depth:
        return results
    
    # Get all attributes of the object
    for attr_name in dir(model):
        # Skip private attributes and methods
        if attr_name.startswith('_'):
            continue
            
        try:
            attr_value = getattr(model, attr_name)
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name
            
            # Check if it's a tensor
            if isinstance(attr_value, torch.Tensor):
                # Check density and sparsity
                is_sparse = attr_value.is_sparse or attr_value.is_sparse_csr
                is_contiguous = attr_value.is_contiguous()
                is_cuda = attr_value.device.type == 'cuda'
                
                device_info = {
                    'name': full_name,
                    'shape': tuple(attr_value.shape),
                    'dtype': str(attr_value.dtype),
                    'device': str(attr_value.device),
                    'requires_grad': attr_value.requires_grad,
                    'is_sparse': is_sparse,
                    'is_contiguous': is_contiguous,
                    'stride': tuple(attr_value.stride()) if not is_sparse else 'N/A (sparse)',
                    'storage_offset': attr_value.storage_offset() if not is_sparse else 'N/A (sparse)',
                    'numel': attr_value.numel()
                }
                
                # Categorize by issues
                has_issues = []
                if not is_cuda:
                    has_issues.append('CPU')
                if is_sparse:
                    has_issues.append('SPARSE')
                    results['sparse_tensors'].append(device_info)
                if not is_contiguous:
                    has_issues.append('NON_CONTIGUOUS')
                    results['non_contiguous_tensors'].append(device_info)
                
                if has_issues:
                    device_info['issues'] = has_issues
                    results['problematic_tensors'].append(device_info)
                
                # Still categorize by device for backward compatibility
                if is_cuda:
                    results['cuda_tensors'].append(device_info)
                else:
                    results['cpu_tensors'].append(device_info)
            
            # Check if it's a Parameter
            elif isinstance(attr_value, nn.Parameter):
                is_sparse = attr_value.is_sparse or attr_value.is_sparse_csr
                is_contiguous = attr_value.is_contiguous()
                is_cuda = attr_value.device.type == 'cuda'
                
                device_info = {
                    'name': full_name,
                    'shape': tuple(attr_value.shape),
                    'dtype': str(attr_value.dtype),
                    'device': str(attr_value.device),
                    'requires_grad': attr_value.requires_grad,
                    'type': 'Parameter',
                    'is_sparse': is_sparse,
                    'is_contiguous': is_contiguous,
                    'stride': tuple(attr_value.stride()) if not is_sparse else 'N/A (sparse)',
                    'storage_offset': attr_value.storage_offset() if not is_sparse else 'N/A (sparse)',
                    'numel': attr_value.numel()
                }
                
                # Categorize by issues
                has_issues = []
                if not is_cuda:
                    has_issues.append('CPU')
                if is_sparse:
                    has_issues.append('SPARSE')
                    results['sparse_tensors'].append(device_info)
                if not is_contiguous:
                    has_issues.append('NON_CONTIGUOUS')
                    results['non_contiguous_tensors'].append(device_info)
                
                if has_issues:
                    device_info['issues'] = has_issues
                    results['problematic_tensors'].append(device_info)
                
                if is_cuda:
                    results['cuda_tensors'].append(device_info)
                else:
                    results['cpu_tensors'].append(device_info)
                    
            # Check if it's a Module (like torchmetrics)
            elif isinstance(attr_value, nn.Module):
                # Get device info for the module
                module_devices = set()
                for param in attr_value.parameters():
                    module_devices.add(param.device.type)
                for buffer in attr_value.buffers():
                    module_devices.add(buffer.device.type)
                
                if len(module_devices) > 1:
                    results['mixed_tensors'].append({
                        'name': full_name,
                        'type': type(attr_value).__name__,
                        'devices': list(module_devices)
                    })
                elif len(module_devices) == 1:
                    device_type = list(module_devices)[0]
                    module_info = {
                        'name': full_name,
                        'type': type(attr_value).__name__,
                        'device': device_type
                    }
                    
                    if device_type == 'cuda':
                        results['cuda_tensors'].append(module_info)
                    else:
                        results['cpu_tensors'].append(module_info)
                
                # Recursively inspect the module
                if current_depth < max_depth - 1:
                    sub_results = inspect_model_devices(attr_value, full_name, max_depth, current_depth + 1)
                    for key in results:
                        results[key].extend(sub_results[key])
            
            # Check for other types that might contain tensors
            elif hasattr(attr_value, '__dict__') and not callable(attr_value):
                if current_depth < max_depth - 1:
                    sub_results = inspect_model_devices(attr_value, full_name, max_depth, current_depth + 1)
                    for key in results:
                        results[key].extend(sub_results[key])
            else:
                # Non-tensor attribute
                if not callable(attr_value):
                    results['non_tensor_attrs'].append({
                        'name': full_name,
                        'type': type(attr_value).__name__,
                        'value': str(attr_value)[:100]  # Truncate long values
                    })
                    
        except Exception as e:
            results['errors'].append({
                'name': full_name,
                'error': str(e)
            })
            continue
    
    return results

def print_device_report(model, detailed=False):
    """
    Print a formatted report of device allocation for all model components.
    
    Args:
        model: PyTorch Lightning model to inspect
        detailed: If True, show detailed information for each tensor
    """
    print("="*80)
    print("COMPREHENSIVE DEVICE & TENSOR DENSITY REPORT")
    print("="*80)
    
    results = inspect_model_devices(model)
    
    # Show problematic tensors first (most important)
    if results['problematic_tensors']:
        print(f"\n🚨 PROBLEMATIC TENSORS ({len(results['problematic_tensors'])}) - LIKELY CAUSING ISSUES!")
        print("-" * 70)
        for item in results['problematic_tensors']:
            issues_str = " | ".join(item['issues'])
            print(f"  ❌ {item['name']}: {issues_str}")
            if detailed:
                print(f"     Shape: {item['shape']} | Device: {item['device']}")
                print(f"     Contiguous: {item['is_contiguous']} | Sparse: {item['is_sparse']}")
                if item['stride'] != 'N/A (sparse)':
                    print(f"     Stride: {item['stride']} | Storage offset: {item['storage_offset']}")
                print()
    
    print(f"\n📍 CUDA TENSORS/MODULES ({len(results['cuda_tensors'])})")
    print("-" * 40)
    for item in results['cuda_tensors']:
        status_indicators = []
        if not item.get('is_contiguous', True):
            status_indicators.append('NON-CONTIGUOUS')
        if item.get('is_sparse', False):
            status_indicators.append('SPARSE')
        
        status_str = f" [{', '.join(status_indicators)}]" if status_indicators else ""
        
        if detailed:
            if 'shape' in item:  # It's a tensor
                print(f"  {item['name']}: {item['shape']} | {item['dtype']} | {item['device']}{status_str}")
                if 'stride' in item and item['stride'] != 'N/A (sparse)':
                    print(f"    ↳ Contiguous: {item.get('is_contiguous', 'N/A')} | Stride: {item['stride']}")
            else:  # It's a module
                print(f"  {item['name']}: {item['type']} | {item['device']}{status_str}")
        else:
            print(f"  ✅ {item['name']}{status_str}")
    
    print(f"\n💻 CPU TENSORS/MODULES ({len(results['cpu_tensors'])})")
    print("-" * 40)
    for item in results['cpu_tensors']:
        status_indicators = []
        if not item.get('is_contiguous', True):
            status_indicators.append('NON-CONTIGUOUS')
        if item.get('is_sparse', False):
            status_indicators.append('SPARSE')
        
        status_str = f" [{', '.join(status_indicators)}]" if status_indicators else ""
        
        if detailed:
            if 'shape' in item:  # It's a tensor
                print(f"  {item['name']}: {item['shape']} | {item['dtype']} | {item['device']}{status_str}")
                if 'stride' in item and item['stride'] != 'N/A (sparse)':
                    print(f"    ↳ Contiguous: {item.get('is_contiguous', 'N/A')} | Stride: {item['stride']}")
            else:  # It's a module
                print(f"  {item['name']}: {item['type']} | {item['device']}{status_str}")
        else:
            print(f"  ❌ {item['name']}{status_str}")
    
    if results['sparse_tensors']:
        print(f"\n🕳️  SPARSE TENSORS ({len(results['sparse_tensors'])})")
        print("-" * 40)
        for item in results['sparse_tensors']:
            print(f"  {item['name']}: {item['shape']} | {item['device']} | SPARSE")
    
    if results['non_contiguous_tensors']:
        print(f"\n📐 NON-CONTIGUOUS TENSORS ({len(results['non_contiguous_tensors'])})")
        print("-" * 40)
        for item in results['non_contiguous_tensors']:
            print(f"  {item['name']}: {item['shape']} | Stride: {item['stride']}")
            if detailed:
                print(f"    ↳ Storage offset: {item['storage_offset']}")
    
    if results['mixed_tensors']:
        print(f"\n⚠️  MIXED DEVICE MODULES ({len(results['mixed_tensors'])})")
        print("-" * 40)
        for item in results['mixed_tensors']:
            print(f"  {item['name']}: {item['type']} | Devices: {item['devices']}")
    
    if results['errors']:
        print(f"\n❗ ERRORS ({len(results['errors'])})")
        print("-" * 40)
        for item in results['errors']:
            print(f"  {item['name']}: {item['error']}")
    
    print(f"\n📊 DETAILED SUMMARY")
    print("-" * 40)
    print(f"  CUDA components: {len(results['cuda_tensors'])}")
    print(f"  CPU components:  {len(results['cpu_tensors'])}")
    print(f"  Sparse tensors:  {len(results['sparse_tensors'])}")
    print(f"  Non-contiguous:  {len(results['non_contiguous_tensors'])}")
    print(f"  Mixed components: {len(results['mixed_tensors'])}")
    print(f"  Errors: {len(results['errors'])}")
    print(f"  🚨 TOTAL PROBLEMATIC: {len(results['problematic_tensors'])}")
    
    if results['problematic_tensors']:
        print(f"\n⚠️  CRITICAL: Found {len(results['problematic_tensors'])} problematic tensors!")
        print("   These are likely causing the 'Tensors must be CUDA and dense' error.")
        print("   Focus on fixing the tensors marked with 🚨 above.")