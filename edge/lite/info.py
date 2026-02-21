VERSIONS = [1,2,3,4,5,6,7, 8]
def get_path (i):
    if i > len(VERSIONS) - 1: 
        return None
    return f"lite-models/model_{VERSIONS[i]}.tflite"
