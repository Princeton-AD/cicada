import h5py

def print_attrs(name, obj):
    print(f"\n{name}")
    for key, val in obj.attrs.items():
        print(f"  Attr - {key}: {val}")

def print_h5_contents(filename):
    with h5py.File(filename, 'r') as f:
        print(f"File: {filename}\n")
        
        # Recursively visit all groups and datasets
        f.visititems(print_attrs)
        
        # Also print dataset shapes and dtype
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                # Optionally print the data itself (commented out here)
                # print(f"  Data:\n{obj[()]}")
        
        f.visititems(print_dataset_info)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python print_h5.py <file.h5>")
    else:
        print_h5_contents(sys.argv[1])

