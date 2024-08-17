import os

def list_files(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def get_stem(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def delete_files(folder_path, stem_list):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_stem = get_stem(file)
            if file_stem not in stem_list:
                file_path = os.path.join(root, file)
                os.remove(file_path)  # Uncomment this line to delete files
                # print(f"Deleted: {file_path}")  # Optional: print the deleted file

folder_path = r'data/metadata'

# Get the list of all files in the metadata folder
file_list = list_files(folder_path)

# Create a list of stems from these files
stem_list = [get_stem(file) for file in file_list]

print("Stems to keep:", stem_list)

# Now delete files from the target folder if their stem is not in the stem_list
delete_files(r'data/images_testing/test_images/input', stem_list)
