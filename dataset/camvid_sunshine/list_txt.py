import os

# Folder paths and output files
folders = [
    ('test/', 'testannot/', 'camvid_test_list.txt'),
    ('valid/', 'validannot/', 'camvid_val_list.txt'),
    ('train/', 'trainannot/', 'camvid_train_list.txt')
]

# Generate individual folders
for image_folder, annotation_folder, output_file in folders:
    # List all image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

    # Write to output file
    with open(output_file, 'w') as f:
        for image in image_files:
            annotation = image.replace('.jpg', '.png')  # Assuming annotation filename matches but with .png extension
            f.write(f'{image_folder}{image} {annotation_folder}{annotation}\n')

    print(f'File {output_file} generated successfully.')

# Generate trainval folder by combining train and valid
train_files = sorted([f for f in os.listdir('train/') if f.endswith('.jpg')])
valid_files = sorted([f for f in os.listdir('valid/') if f.endswith('.jpg')])

with open('camvid_trainval_list.txt', 'w') as f:
    for image in train_files:
        annotation = image.replace('.jpg', '.png')
        f.write(f'train/{image} trainannot/{annotation}\n')
    for image in valid_files:
        annotation = image.replace('.jpg', '.png')
        f.write(f'valid/{image} validannot/{annotation}\n')

print('File camvid_trainval.txt generated successfully.')

