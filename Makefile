## Create raw data folder & put product codes there
dataset:
	# Create a directory for storing raw data, including the
	# parent path and without raising an error if this
	# directory already exists.
	mkdir -p data/raw

	# Download the product classifications from Harvard CID
	curl https://intl-atlas-downloads.s3.amazonaws.com/classifications.xlsx --output data/raw/classifications.xlsx