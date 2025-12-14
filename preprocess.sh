# Download Training Split and Validation Split of CC3M dataset from here: https://ai.google.com/research/ConceptualCaptions/download
# Then run sed -i '1s/^/caption\turl\n/' Train_GCC-training.tsv if you are on Linux or sed -i '' '1s/^/caption\turl\n/' Train_GCC-training.tsv if you are on MacOS to add header to the tsv file.
# Then run the following command to create webdataset format of CC3M dataset. Before that, install img2dataset by running pip install img2dataset.
# Remember to adjust the processes_count and thread_count according to your machine configuration.
img2dataset --url_list ./data/Train_GCC-training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb False
