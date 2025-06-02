import os
import gdown
import zipfile
import re
import shutil
import tempfile

def download_and_unzip_gdrive_file(gdrive_url: str, destination_dir: str):
    """
    Downloads a file from a Google Drive URL, and moves its contents to the destination directory.

    Args:
        gdrive_url (str): The Google Drive URL of the file to download.
        destination_dir (str): The directory where the file should be unzipped.
    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"Starting download & extraction: {gdrive_url} -> {destination_dir}")

    os.makedirs(destination_dir, exist_ok=True)

    file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', gdrive_url)
    if not file_id_match:
        print(f"Error: Could not extract file ID from URL: {gdrive_url}")
        return False
    file_id = file_id_match.group(1)

    zip_filename = "downloaded_gdrive_file.zip"
    output_path = os.path.join(destination_dir, zip_filename)

    try:
        gdown.download(id=file_id, output=output_path, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"Error during gdown.download for ID {file_id}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print(f"Error: Download failed or file is empty: {output_path}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    
    downloaded_size = os.path.getsize(output_path)
    print(f"Downloaded {output_path} ({downloaded_size} bytes).")

    temp_dir_for_extraction = tempfile.mkdtemp()

    try:
        print(f"Unzipping {output_path} to temporary directory {temp_dir_for_extraction}...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir_for_extraction)
        print(f"Successfully unzipped {output_path} to {temp_dir_for_extraction}.")

        # Determine the source of files to move
        extracted_items = os.listdir(temp_dir_for_extraction)
        source_path_for_moving = temp_dir_for_extraction

        if len(extracted_items) == 1:
            potential_single_folder = os.path.join(temp_dir_for_extraction, extracted_items[0])
            if os.path.isdir(potential_single_folder):
                source_path_for_moving = potential_single_folder
        
        # Ensure final destination directory exists
        os.makedirs(destination_dir, exist_ok=True)

        print(f"Processing and moving files from '{source_path_for_moving}' to '{destination_dir}'...")
        
        files_moved_count = 0
        # os.walk will iterate through all files and directories in source_path_for_moving
        for root, _, files_in_dir in os.walk(source_path_for_moving):
            for filename in files_in_dir:
                src_file_full_path = os.path.join(root, filename)
                
                # Determine the path of the file relative to the source_path_for_moving
                # This relative path will be used to construct the destination path
                relative_path_to_file = os.path.relpath(src_file_full_path, source_path_for_moving)
                dst_file_full_path = os.path.join(destination_dir, relative_path_to_file)
                
                # Ensure the parent directory for the destination file exists
                dst_file_parent_dir = os.path.dirname(dst_file_full_path)
                os.makedirs(dst_file_parent_dir, exist_ok=True)

                if os.path.isfile(dst_file_full_path):
                    print(f"Warning: Overwriting existing file '{dst_file_full_path}'.")

                shutil.move(src_file_full_path, dst_file_full_path)
                files_moved_count += 1
        
        if files_moved_count > 0:
            print(f"Successfully moved {files_moved_count} file(s) to {destination_dir}.")
        else:
            # Provide a more specific note if no files were moved.
            if not extracted_items: # Nothing was extracted from the zip initially
                 print(f"Note: The zip file '{os.path.basename(output_path)}' appears to be completely empty.")
            elif source_path_for_moving != temp_dir_for_extraction and not os.listdir(source_path_for_moving):
                 # This means a single root folder was identified, and it was empty.
                 print(f"Note: The single root folder '{os.path.basename(source_path_for_moving)}' (from zip) was empty, so no files were moved.")
            else: # Zip either contained only empty directories, or the structure didn't yield files from source_path_for_moving
                 print(f"Note: No files found to move from '{source_path_for_moving}'. The zip may have contained only empty directories.")
        
        # If we reached here, unzipping and moving (or handling of empty zip) is considered successful path.
        # Now, remove the original zip file.
        os.remove(output_path)
        
        return True # Overall success for this function call

    except zipfile.BadZipFile:
        print(f"Error: File {output_path} (size: {downloaded_size} bytes) is not a valid zip file.")
        if os.path.exists(output_path): # Attempt to remove the bad zip
            os.remove(output_path)
            print(f"Removed invalid zip: {output_path}.")
        return False # Function call failed
    except Exception as e_unzip:
        # This catches other errors during unzipping or the file moving logic.
        print(f"Error during unzipping or moving of {output_path}: {e_unzip}")
        if os.path.exists(output_path): # Attempt to remove the zip if an error occurred
            os.remove(output_path)
            print(f"Removed zip after error: {output_path}.")
        return False # Function call failed
    finally:
        # Always try to clean up the temporary extraction directory
        if os.path.exists(temp_dir_for_extraction):
            print(f"Cleaning up temporary extraction directory: {temp_dir_for_extraction}")
            shutil.rmtree(temp_dir_for_extraction)


if __name__ == "__main__":
    data_files = {
        "best_returns_teammates": {
            "gdrive_url": "https://drive.google.com/file/d/1pS0wvJDzOZa954RADF_j9ETR74THzh8I/view?usp=sharing",
            "target_directory": "results/"
        },
        "eval_teammates": {
            "gdrive_url": "https://drive.google.com/file/d/1KjBV2GekKdRBiK6QSGe2vYx2ThXlG7X7/view?usp=sharing",
            "target_directory": "eval_teammates/"
        },
    }

    for data_name, data_info in data_files.items():
        success = download_and_unzip_gdrive_file(gdrive_url=data_info["gdrive_url"], destination_dir=data_info["target_directory"])

        if success:
            print(f"Download completed successfully for {data_name}.")
        else:
            print(f"Download failed for {data_name}.")
