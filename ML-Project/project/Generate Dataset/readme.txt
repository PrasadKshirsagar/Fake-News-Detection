--------------------------------------------ABOUT-------------------------------------------------------
This folder contains the code to generate the dataset to be used in the models.

'process1.py' generates 'Dataset 1'. It reads 'articles1.csv' and 'fake.csv' and generates 'dataset1.txt' and 'dataset2.txt' in the 'Dataset1' folder.

'process2.py' generates 'Dataset 2'. It reads 'dataset1.txt' and 'dataset2.txt' and generates 'NEW_myX' and 'NEW_myX_test' in the 'Dataset2' folder.

Compatible with python3 in linux

Dependencies:
->	nltk
->	pandas
->	collections
->	Decimal

use pip install dependency to install the missing dependency.

Run as 

	python3 process1.py
	python3 process2.py


