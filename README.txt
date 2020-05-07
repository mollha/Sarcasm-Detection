
List of dependencies:
-



Datasets can be collected from the following sources:
Twitter data - Ptáček et al. (2014):
- Collected from: http://liks.fav.zcu.cz/sarcasm/
- Uses the EN balanced corpus containing 100,000 tweet IDs that must be scraped from the Twitter API - Twitter scraper can be found in /pkg/datasets/ptacek/processing_scrips/TwitterCrawler
- Once downloaded, move normal.txt and sarcastic.txt (files from download) into /pkg/datasets/ptacek/raw_data
  e.g. /pkg/datasets/ptacek/raw_data
                               ├── /normal.txt
                               ├── /sarcastic.txt

News headlines - Misra et al. (2019):
- Data is downloaded in JSON format
- https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection
- Once downloaded, move Sarcasm_Headlines_Dataset_v2.json (files from download) into /pkg/datasets/news_headlines/raw_data
  e.g. /pkg/datasets/news_headlines/raw_data
                               ├── /Sarcasm_Headlines_Dataset_v2.json


Amazon reviews - Filatova et al. (2012):
- Data is downloaded in .rar format
- https://github.com/ef2020/SarcasmAmazonReviewsCorpus/
- Only Ironic.rar and Regular.rar is used in this project
- Convert Ironic.rar and Regular.rar (files from download) into regular folders, then move them to /pkg/datasets/amazon_reviews/raw_data
  e.g. /pkg/datasets/news_headlines/raw_data
                                      ├── /Ironic
                                      ├── /Regular


After downloading the data - proceed to reformat it into a csv and apply our data cleaning processes:
- Run p1_create_raw_csv.py followed by p2_clean_original_data.py to achieve the correct configuration => NOTE: p1_create_raw_csv.py will take some time on the Twitter dataset, as it is slow to scrape the Tweets given their ids
  e.g. /pkg/datasets/news_headlines/
                        ├── /processed_data
                                ├── ...
                                ├── /CleanData.csv
                                ├── /OriginalData.csv
                        ├── /processing_scripts/...
                        ├── /raw_Data/...