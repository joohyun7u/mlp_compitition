from shutil import copyfile


copyfile('/local_datasets/MLinP/train/clean/HackSpace51.00097.02.tif','./output/clean_1.tif')
copyfile('/local_datasets/MLinP/train/clean/HackSpace51.00094.01.tif','./output/clean_2.tif')

copyfile('/local_datasets/MLinP/train/scan/HackSpace51.00097.02.tif','./output/scan_1.tif')
copyfile('/local_datasets/MLinP/train/scan/HackSpace51.00094.01.tif','./output/scan_2.tif')

copyfile('/local_datasets/MLinP/test/scan/BookOfMaking.00012.02.tif','./output/test_1.tif')
copyfile('/local_datasets/MLinP/test/scan/BookOfMaking.00014.01.tif','./output/test_2.tif')

