#!/bin/sh
# the data directory where we will store the conll-2012 data
DATA_DIR="$1"

# throw an error if the data directory does not exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: data directory '$DATA_DIR' does not exist."
    exit 1
fi

# throw an error if the target directory already exists
if [ -d "$DATA_DIR/conll-2012" ]; then
    echo "Error: target directory '$DATA_DIR/conll-2012' already exists."
    exit 1
fi

# move into the target directory
cd "$DATA_DIR" || exit 1

echo "Downloading the data..."
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zmycy7t9h9-2.zip

echo "Unpacking the data..."
unzip zmycy7t9h9-2.zip
unzip zmycy7t9h9-2/conll-2012.zip

echo "Removing the zip file and temporary directory..."
rm zmycy7t9h9-2.zip
rm -rf zmycy7t9h9-2
