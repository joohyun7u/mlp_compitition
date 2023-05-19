import os
import zipfile

os.makedirs('/local_datasets/MLinP/train/clean')
os.makedirs('/local_datasets/MLinP/train/scan')
os.makedirs('/local_datasets/MLinP/test/scan')
    	
train_clean_zip = zipfile.ZipFile('./train_clean.zip')
train_clean_zip.extractall('/local_datasets/MLinP/train/clean')
train_clean_zip.close()

train_scan_zip = zipfile.ZipFile('./train_scan.zip')
train_scan_zip.extractall('/local_datasets/MLinP/train/scan')
train_scan_zip.close()

test_scan_zip = zipfile.ZipFile('./test_scan.zip')
test_scan_zip.extractall('/local_datasets/MLinP/test/scan')
test_scan_zip.close()