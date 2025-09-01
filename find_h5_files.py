import os

def find_h5_files(directory: str):
    # Walk through the directory and its subdirectories
    for dirpath, _, filenames in os.walk(directory):
        # Check if any file ends with '.h5'
        for filename in filenames:
            if filename.endswith('.h5'):
                print(f"Found .h5 file: {os.path.join(dirpath, filename)}")

# Example usage
directory = "/hdfs/store/user/aloelige/TT_TuneCP5_13p6TeV_powheg-pythia8/"
find_h5_files(directory)