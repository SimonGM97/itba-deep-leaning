from PyTradeX.config.params import Params
import os


def sincronize_file_system_buckets(
    source_bucket: str, 
    destination_bucket: str, 
    sub_dir: str = None,
    debug: bool = False
):
    """
    All objects and file structure in the destination_bucket will be updated, copying the objects and
    file structure found in the destination_bucket.
    """
    def find_objects(bucket: str):
        objects = set()
        keys_dict = {}

        for root, directories, files in os.walk(os.path.join(Params.base_cwd, bucket)):
            for file in files:
                key = os.path.join(root, file)
                size = os.path.getsize(key)
                last_modified = os.path.getmtime(key)
                if '.DS_Store' not in file:
                    search_name = key.split(bucket)[1]
                    objects.add((search_name, size, last_modified))
                    keys_dict[search_name] = key
        return objects, keys_dict
    
    # Find Bucket Objects
    source_objects, source_keys = find_objects(source_bucket)
    destination_objects, destination_keys = find_objects(destination_bucket)

    # Find objects to remove from destination bucket
    remove_objects = destination_objects - source_objects
    
    # Remove objects
    for obj in remove_objects:
        if debug:
            print(f"Removing: {destination_keys[obj[0]]}")

        os.remove(destination_keys[obj[0]])

    # Find objects to add to destination bucket
    add_objects = source_objects - destination_objects

    # Add objects
    for obj in add_objects:
        source_key = source_keys[obj[0]]
        destination_key = source_key.replace(source_bucket, destination_bucket)
        obj_sub_dir = destination_key[:destination_key.rfind('/') + 1]

        # Create sub_dir if it does not exist in destination bucket
        if not os.path.exists(obj_sub_dir):
            print(f"Creating {obj_sub_dir} in {destination_bucket}.")
            os.makedirs(obj_sub_dir)

        with open(source_key, "rb") as source_file:
            with open(destination_key, "wb") as destination_file:
                if debug:
                    print(f"Adding {destination_key}.")
                    
                destination_file.write(source_file.read())
