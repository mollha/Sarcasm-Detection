# Machine Learning and Deep Learning Approaches to Sarcasm Detection

This project addresses the problem of sarcasm detection - often quoted as a subtask of sentiment analysis. There are two main scripts used to begin using this code:
- **train.py** : trains and evaluates new models on chosen dataset, saving these models to [Code/pkg/trained_models/](../../tree/master/Code/pkg/trained_models/)
- **console.py** : makes predictions using existing trained models, where user input can be provided via a console. A visualisation of attention weights is produced in /colorise.html




### List of dependencies:

- [en-core-web-md](https://spacy.io/models/en) == 2.2.5
- [Keras](https://keras.io/) == 2.3.1
- [matplotlib](https://matplotlib.org/) == 3.1.3
- [numpy](https://numpy.org/) == 1.18.0
- [pandas](https://pandas.pydata.org/) == 0.25.3
- [pycorenlp](https://pypi.org/project/pycorenlp/) == 0.3.0
- [scikit-learn](https://scikit-learn.org/stable/) == 0.22
- [spacy](https://spacy.io/) == 2.2.3
- [tensorflow](https://www.tensorflow.org/) == 2.1.0
- [tensorflow-hub](https://www.tensorflow.org/hub) == 0.7.0
- [tweepy](https://www.tweepy.org/) == 3.8.0
- [twitter](https://pypi.org/project/python-twitter/) == 1.18.0

### Data configuration and Setup

**Datasets can be collected from the following sources:**

- Twitter data - Ptáček et al. (2014):
    - Collected from: http://liks.fav.zcu.cz/sarcasm/
    - Uses the EN balanced corpus containing 100,000 tweet IDs that must be scraped from the Twitter API - Twitter scraper can be found in Code/pkg/datasets/ptacek/processing_scrips/TwitterCrawler
    - Once downloaded, move normal.txt and sarcastic.txt (files from download) into Code/pkg/datasets/ptacek/raw_data\

- News headlines - Misra et al. (2019):
    - Data is downloaded in JSON format
    - https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
    - Once downloaded, move Sarcasm_Headlines_Dataset_v2.json (files from download) into Code/pkg/datasets/news_headlines/raw_data


- Amazon reviews - Filatova et al. (2012):
    - Data is downloaded in .rar format
    - https://github.com/ef2020/SarcasmAmazonReviewsCorpus/
    - Only Ironic.rar and Regular.rar is used in this project
    - Convert Ironic.rar and Regular.rar (files from download) into regular folders, then move them to Code/pkg/datasets/amazon_reviews/raw_data


**After downloading the data - proceed to reformat it into a csv and apply our data cleaning processes**:
- Run p1_create_raw_csv.py followed by p2_clean_original_data.py to achieve the correct configuration => NOTE: p1_create_raw_csv.py will take some time on the Twitter dataset, as it is slow to scrape the Tweets given their ids\
  e.g. Code/pkg/datasets/news_headlines/\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── /processed_data\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── ...\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── /CleanData.csv\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── /OriginalData.csv\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── /processing_scripts/...\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── /raw_Data/...
   
   
