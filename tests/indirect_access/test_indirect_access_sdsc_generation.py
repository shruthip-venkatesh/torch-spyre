#!/usr/bin/env python3
"""
Demonstration of indirect access compilation pipeline.

This script:
1. Creates input and index tensors
2. Converts indices to addresses
3. Generates SDSC JSON with indirect access fields
4. Compiles with dxp_standalone

Flow:
  Tensors -> OpSpec -> SDSC JSON -> dxp_standalone -> Compiled Kernel
"""

import torch
import json
import subprocess
import os
import sys

def create_indirect_access_opspec():
    """Create an OpSpec for indirect access operation."""
    from sympy import Symbol
    from torch_spyre._C import DataFormats
    from torch_spyre._inductor.op_spec import OpSpec, TensorArg
    
    # Define dimensions
    batch_dim = Symbol("batch")
    feature_dim = Symbol("feature")
    
    # Input tensor (value tensor) - accessed indirectly
    input_tensor = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[128, 64],
        device_coordinates=[batch_dim, feature_dim],
        allocation=None,
        is_index_tensor=False,
        related_value_tensor_idx=-1
    )
    
    # Index tensor - contains indices for indirect access
    # Must have same dimensionality as output for SDSC generation
    index_tensor = TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=DataFormats.SENUINT32,
        device_size=[32, 64],  # Match output dimensions
        device_coordinates=[batch_dim, feature_dim],
        allocation=None,
        is_index_tensor=True,  # Mark as index tensor
        related_value_tensor_idx=0  # Links to input_tensor
    )
    
    # Output tensor
    output_tensor = TensorArg(
        is_input=False,
        arg_index=2,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[32, 64],
        device_coordinates=[batch_dim, feature_dim],
        allocation=None,
        is_index_tensor=False,
        related_value_tensor_idx=-1
    )
    
    # Create OpSpec
    op_spec = OpSpec(
        op="identity",  # Using identity as base operation
        is_reduction=False,
        iteration_space={
            batch_dim: (32, 2),  # 32 elements, split across 2 cores
            feature_dim: (64, 1),  # 64 features, no split
        },
        args=[input_tensor, index_tensor, output_tensor],
        op_info={"indirect_access": True}
    )
    
    return op_spec


def generate_sdsc_bundle(op_spec, output_dir):
    """Generate SDSC bundle from OpSpec using bundle.py."""
    from torch_spyre._inductor.codegen.bundle import generate_bundle
    
    print("\n2. Generating SDSC bundle...")
    
    # Generate bundle (creates sdsc_0.json and bundle.mlir)
    generate_bundle("indirect_add_kernel", output_dir, [op_spec])
    print(f"   ✓ Bundle generated in: {output_dir}/")
    
    # Read the generated SDSC JSON for inspection
    sdsc_file = os.path.join(output_dir, "sdsc_0.json")
    with open(sdsc_file, 'r') as f:
        sdsc_json = json.load(f)
    print(f"   ✓ SDSC JSON: {sdsc_file}")
    print(f"   ✓ Bundle MLIR: {os.path.join(output_dir, 'bundle.mlir')}")
    
    # Print the JSON for inspection
    # print("\n   Generated SDSC JSON:")
    # print("   " + "-" * 76)
    # json_str = json.dumps(sdsc_json, indent=2)
    # for line in json_str.split('\n')[:50]:  # Print first 50 lines
    #     print(f"   {line}")
    # if len(json_str.split('\n')) > 50:
    #     print(f"   ... ({len(json_str.split('\n')) - 50} more lines)")
    # print("   " + "-" * 76)
    
    return sdsc_json

def compile_with_dxp(output_dir):
    """Compile SDSC bundle with dxp_standalone."""
    print("\n4. Compiling with dxp_standalone...")
    
    try:
        # Run dxp_standalone
        cmd = ["dxp_standalone", "--bundle", "-d", output_dir]
        print(f"   Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("   ✓ Compilation successful!")
            print(f"   Output:\n{result.stdout}")
            return True
        else:
            print(f"   ✗ Compilation failed with code {result.returncode}")
            print(f"   Error:\n{result.stderr}")
            return False
            
    except FileNotFoundError:
        print("   ✗ dxp_standalone not found in PATH")
        print("   Please ensure DeepTools is installed and dxp_standalone is accessible")
        return False
    except subprocess.TimeoutExpired:
        print("   ✗ Compilation timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def main():
    print("=" * 80)
    print("Indirect Access Compilation Pipeline Demo")
    print("=" * 80)
    
    # Step 1: Create OpSpec
    print("\n1. Creating OpSpec with indirect access...")
    try:
        op_spec = create_indirect_access_opspec()
        print("   ✓ OpSpec created")
        print(f"   - Operation: {op_spec.op}")
        print(f"   - Args: {len(op_spec.args)}")
        
        # Check indirect access metadata
        for i, arg in enumerate(op_spec.args):
            if arg.is_index_tensor:
                print(f"   - Arg {i}: INDEX TENSOR (links to arg {arg.related_value_tensor_idx})")
            else:
                print(f"   - Arg {i}: {'INPUT' if arg.is_input else 'OUTPUT'} tensor")
                
    except Exception as e:
        print(f"   ✗ Error creating OpSpec: {e}")
        print("\n   Please rebuild torch-spyre:")
        print("     pip install -e .")
        return False
    
    # Step 2 & 3: Generate SDSC JSON
    # Create output directory in current directory
    output_dir = "indirect_access_bundle"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nUsing output directory: {output_dir}/")
    
    try:
        sdsc_json = generate_sdsc_bundle(op_spec, output_dir)
        sdsc_file = os.path.join(output_dir, "sdsc_0.json")
        
        # Verify indirect access fields
        print("\n3. Verifying indirect access fields...")
        op_name = list(sdsc_json.keys())[0]
        dscs = sdsc_json[op_name]["dscs_"][0][op_name]
        
        # Check allocations
        schedule_tree = dscs["scheduleTree_"]
        has_indirect = False
        value_tensor_found = False
        index_tensor_found = False
        
        for node in schedule_tree:
            if node["nodeType_"] == "allocate":
                alloc_type = node.get("indirectAllocType_", "NOT_FOUND")
                if alloc_type == "value_tensor":
                    value_tensor_found = True
                    print(f"   ✓ Found value_tensor: {node['name_']}")
                elif alloc_type == "index_tensor":
                    index_tensor_found = True
                    related = node.get("relatedIndirectAccessAlloc_", "NOT_FOUND")
                    print(f"   ✓ Found index_tensor: {node['name_']}")
                    print(f"     - Related to: {related}")
                    has_indirect = True
        
        # Check labeled DS
        labeled_ds = dscs["labeledDs_"]
        for lds in labeled_ds:
            if lds.get("dsType_") == "KERNEL_IDX":
                print(f"   ✓ Found KERNEL_IDX: {lds['dsName_']}")
        
        # Check compute op
        compute_ops = dscs["computeOp_"]
        for op in compute_ops:
            if "indirectAccessIndexLabeledDs" in op:
                print(f"   ✓ Found indirectAccessIndexLabeledDs: {op['indirectAccessIndexLabeledDs']}")
        
        if not has_indirect:
            print("   ✗ WARNING: No indirect access fields found!")
            return False
        
        if not value_tensor_found:
            print("   ✗ WARNING: No value_tensor found!")
            return False
        
        if not index_tensor_found:
            print("   ✗ WARNING: No index_tensor found!")
            return False
    except Exception as e:
        print(f"   ✗ Error generating SDSC: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Compile with dxp_standalone
    print("\n4. Compiling with dxp_standalone...")
    success = compile_with_dxp(output_dir)
    
    if success:
        # List generated artifacts
        print("\n5. Compilation successful! Generated artifacts:")
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"   - {item} ({size} bytes)")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
