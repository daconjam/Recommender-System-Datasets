# Recommeder System Datasets 
This repository contains a list of public and compatible datasets, noting other major repositories containing newer, and popular real-world datasets that are available, along with reference of sample code for respective recomendation tasks. Most of the datasets presented are for non-commercial use by academics, for example faculty, university researchers and
other scientists. The datasets are free, however datasets may ask for citation. 
>In addition, there are a few links that may contain some sample code from existing works by their respective author. Before using these datasets, please review their sites and/ or **README** files for their respective usage licenses, acknowledgments and other details as a few datasets have additional citation requests. These requests can be found on the bottom of each dataset's web page.

## Contributors

    Name: Jamell Dacon
    Email: daconjam at msu dot edu (daconjam@msu.edu)
    
If you publish material based on material and/ or information obtained from this repository, then, in your acknowledgements, please note the assistance you received from utilizing this repository. By citing our paper as follows below, feel free to star ![GitHub stars](https://img.shields.io/github/stars/daconjam/Recommender-System-Datasets?style=social) and/ or fork ![Github forks](https://img.shields.io/github/forks/daconjam/Recommender-System-Datasets?style=social)
the repository so that academics i.e. university researchers, faculty and other scientists may have quicker access to the available datasets. This will aid in directing others in obtaining the same datasets, thus allowing the replication and improvement of experiments.


## Addition Information: Correspondence

Personal Page: [Portfolio](https://www.cse.msu.edu/~daconjam/)

Lab Page: [DSELab@MSU](https://www.dse.cse.msu.edu)


### Citation

Here is a BiBTeX citation:
    
    @inbook{10.1145/3442442.3452325, author = {Dacon, Jamell and Liu, Haochen}, title = {Does Gender Matter in the News? Detecting and Examining Gender Bias in News Articles}, year = {2021}, isbn = {9781450383134}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, url = {https://doi.org/10.1145/3442442.3452325}, abstract = {To attract unsuspecting readers, news article headlines and abstracts are often written with speculative sentences or clauses. Male dominance in the news is very evident, whereas females are seen as “eye candy” or “inferior”, and are underrepresented and under-examined within the same news categories as their male counterparts. In this paper, we present an initial study on gender bias in news abstracts in two large English news datasets used for news recommendation and news classification. We perform three large-scale, yet effective text-analysis fairness measurements on 296,965 news abstracts. In particular, to our knowledge we construct two of the largest benchmark datasets of possessive (gender-specific and gender-neutral) nouns and attribute (career-related and family-related) words datasets1 which we will release to foster both bias and fairness research aid in developing fair NLP models to eliminate the paradox of gender bias. Our studies demonstrate that females are immensely marginalized and suffer from socially-constructed biases in the news. This paper individually devises a methodology whereby news content can be analyzed on a large scale utilizing natural language processing (NLP) techniques from machine learning (ML) to discover both implicit and explicit gender biases. }, booktitle = {Companion Proceedings of the Web Conference 2021}, pages = {385–392}, numpages = {8} }


## Major repositories with several datasets 

  - [Arizona State University: Social Computing Data Repository](http://socialcomputing.asu.edu/pages/datasets) 
    
   >Note: ASU Social Computing Data Respository contains several Network Datasets
   
  - [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
  - [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)
  - [Yahoo Research Webscope Datasets](https://webscope.sandbox.yahoo.com/)
  
  >  Note: Yahoo Research Ratings and Classification Data [Music, Movies, Tags, Clicks, Images & Videos](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r): This sets of datasets contains music ratings, movie ratings, popular URLs and tags, click log dataset, face images of celebrities and 22K videos. 
  
  - [Kaggle Datasets](https://www.kaggle.com/datasets)
  
  - [GroupLens Datasets](https://grouplens.org/datasets)
  
  - [Recommnder Systems Datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html).  Contributors: [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/)
  
  
## Datasets links via Catergories

The following datasets are very popular in **Recommender Systems**, below are also brief dataset descriptions.

### News 
  - [MIND](https://msnews.github.io/) dataset was collected from the Microsoft News website, for more detailed information about the MIND dataset, you can refer to the following paper: [MIND paper, (Wu et al., 2020)](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf). They randomly sampled news from from October 12 to November 22, 2019 for 6 weeks creating two datasets i.e., MIND and MIND-small both totalling in 161,013 news articles. Each news article contains a news ID, a category label, a title, and a body (url); however, not every article contains an abstract resulting in 96,112 abstracts. We used the training set (largest set of news articles) since both the validation and test sets are subsets of the training set. MIND is created to serve as a new news recommendation benchmark dataset.

   - [NCD](https://www.kaggle.com/rmisra/news-category-dataset) dataset was collected from [Huffpost](https://www.huffpost.com/). The news articles were sampled from news headlines from the year 2012 to 2018 totalling in 202,372 news articles. Each news article contains a category label, headline, authors, link, and date; however, not every article contains a short description (abstract) resulting in 200,853 abstracts. NCD serves as a news classification and recommendation benchmark dataset.
   
   - [ANTCD](https://www.kaggle.com/amananandrai/ag-news-classification-dataset?select=train.csv) dataset was collected by [Zhang et al.](https://arxiv.org/pdf/1509.01626.pdf) from over 2000 news sources by ComeToMyHead (an online academic news search engine) for a under 2 years of activity. They access the original [AG's News Corpus](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) which contained 496,835 news articles, and by choosing the 4 categories with largest samples (30,000 articles each), thus creating the ANTCD Dataset with 120,000 news articles. Each news article contains a category (class index), a title and an abstract. We used the training set (largest set of news articles) since the test set is a subset that only contains 7600 testing samples. ANTCD serves as a news classification and recommendation benchmark dataset.



### E-commerce
  - [Amazon](http://jmcauley.ucsd.edu/data/amazon/): This Amazon dataset consists of reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs) spanning from May 1996 to July 2014.
  - [Amazon - Ratings (Beauty Products)](https://www.kaggle.com/skillsmuggler/amazon-ratings): This is a dataset related to over 2 Million customer reviews and ratings of Beauty related products sold on their website.
  - [Toy Products on Amazon](https://www.kaggle.com/PromptCloudHQ/toy-products-on-amazon): This is a pre-crawled dataset, taken as subset of a bigger [dataset (more than 115k products)](https://datastock.shop) that was created by extracting data from Amazon.com.
  - [Slashdot](https://snap.stanford.edu/data/soc-Slashdot0902.html): The network cotains friend/foe links between the users of Slashdot which was obtained in February 2009.
  - [Taobao](https://tianchi.aliyun.com/datalab/dataSet.htm?spm=5176.100073.888.13.62f83f62aOlMEI&id=1): This dataset contains anonymized users' shopping logs in the past 6 months before and on the "Double 11" day,and the label information indicating whether they are repeated buyers. Due to privacy issue, data is sampled in a biased way, so the statistical result on this data set would deviate from the actual of Tmall.com.
  - Microsoft Web Data [Dataset](https://archive.ics.uci.edu/ml/datasets/Anonymous+Microsoft+Web+Data): This dataset contains a log of anonymous users of www.microsoft.com; with the task predict areas of the web site a user visited based on data on other areas the user visited.
  - [Retailrocket recommender system dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset): This dataset consists of three files: a file with behaviour data (events.csv), a file with item properties (item_properties.сsv) and a file, which describes category tree (category_tree.сsv). The data has been collected from a real-world ecommerce website. 
  - [Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download#English-language_Wikipedia): Wikipedia offers free copies of all available content to interested users. These databases can be used for mirroring, personal use, informal backups, offline use or database queries.
  - [Airbnb Collection](https://www.kaggle.com/fermatsavant/airbnb-dataset-of-barcelona-city): The data was take of http://tomslee.net/airbnb-data-collection-get-the-data, this represent a response of the Barcelona City. The data is collected from the public Airbnb web site without logging in and the code was use is available on https://github.com/tomslee/airbnb-data-collection.
  
  
### Social
  - [Yelp](https://www.yelp.com/dataset): This Yelp dataset is a subset of businesses, reviews, and user-generated data for personal, educational, and academic purposes. This dataset is available in both JSON and SQL files, which can use it to teach students about databases, to learn NLP, or for sample production data while you learn how to make mobile apps.
  - [Facebook](https://www.kaggle.com/sheenabatra/facebook-data): This dataset contains exploratory data analysis that gives insights from a Facebook dataset which consists of identifying users that can be focused more to increase the business. These valuable insights should help Facebook to take intelligent decision to identify its useful users and provide correct recommendations to them.
  - [Twitter](https://snap.stanford.edu/data/ego-Twitter.html): This dataset consists of 'circles' (or 'lists') from Twitter. Twitter data was crawled from public sources. The dataset includes node features (profiles), circles, and ego networks.
  - [Pinterest](https://github.com/kang205/STL-Dataset): This dataset contains the scene-product pairs for fashion and home, respectively.
  

### Stock
  - [Spanish Stocks Historical Data from 2000 to 2019](https://www.kaggle.com/alvarob96/spanish-stocks-historical-data): This dataset contains retrieved retrieve historical data from the companies that integrate the Continuous Spanish Stock Market. May have to refer [investpy](https://github.com/alvarob96/investpy) from Investing.com
  - [Stock Exchange](https://archive.ics.uci.edu/ml/datasets/Machine+Learning+based+ZZAlpha+Ltd.+Stock+Recommendations+2012-2014): This dataset is the ZZAlphaÂ® machine learning recommendations made for various US traded stock portfolios the morning of each day during the 3 year period Jan 1, 2012 - Dec 31, 2014.
  
  
### Job 
  - [Job Recommendation](https://www.kaggle.com/irfanalidv/suggectedjob): This dataset contains a list of recommended jobs listed for individual.
  - [Job Recommendation Analysis](https://www.kaggle.com/kandij/job-recommendation-datasets): A recommendation engine which is build using NLTK helping the applicants to choose thier preferred job based on their application. You will learn how lemmetizer, stemming and vectoriztion are used to process the data and have a better output.
  
 
### Item reviews
  - [Item Learning](https://grouplens.org/datasets/learning-from-sets-of-items-2019/): A dataset for Learning from Sets of Items in Recommender Systems (2019)
  - [eCommerce Item Dataset](https://www.kaggle.com/cclark/product-item-data): This dataset contains 500 actual SKUs from an outdoor apparel brand's product catalog.
  - [Epinions](http://www.trustlet.org/epinions.html): Epinions is a website where people can review products where users can register for free and start writing subjective reviews about many different types of items.
  

### Book
  - [Good Reads](https://www.kaggle.com/jealousleopard/goodreadsbooks): This dataset's purpose is for the requirement of a good clean dataset of books.
  - [Book Crossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/): The BookCrossing (BX) dataset was collected by Cai-Nicolas in a 4-week crawl (August / September 2004) from the Book-Crossing community.
  
### Map
  - [Open OSM](https://planet.openstreetmap.org/planet/full-history/): This data is from OpenStreetMap which is a collaborative mapping project, sort of like Wikipedia but for maps. For reference of python, a few scripts are available at [Hermes repo].(https://github.com/lab41/hermes)
  
  
### Dating
  - [Dating Agency](http://www.occamslab.com/petricek/data/): This dataset contains 17,359,346 anonymous ratings of 168,791 profiles made by 135,359 LibimSeTi users as dumped on April 4, 2006.


### Personality 
  - [Personality 2018](https://grouplens.org/datasets/personality-2018/): The purpose of this dataset is for “User personality and user satisfaction with recommender systems".
  - [DEAPdataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html): This is a dataset for emotion analysis using eeg, physiological and video signals.
  - [MyPersonalityDataset](https://github.com/Myoungs/myPersonality-dataset): This dataset contains information from a popular Facebook application that allowed users to take real psychometric tests, and allowed their Facebook profiles and psychological responses to be recorded (with consent!). Currently, the database contains more than 6,000,000 test results, together with more than 4,000,000 individual Facebook profiles.
  

### Music
  - [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/): The Million Song Dataset is a freely-available collection of audio features and metadata for a million contemporary popular music tracks. For code for the dataset, refer to [MSongDB repo](https://github.com/tbertinmahieux/MSongsDB).
  - [LastFM (Implicit)](https://grouplens.org/datasets/hetrec-2011/): This dataset contains social networking, tagging, and music artist listening information from a set of users from Last.fm online music system, consisting of 92,800 artist listening records from 1892 users.
  
  
### Movies
  - [Netflix](http://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a): This Netflix dataset is the official dataset that was used in the Netflix Prize competition. 
  - [MovieLens](https://grouplens.org/datasets/movielens/): GroupLens Research has collected and made available rating datasets from their movie web site consisting of 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags.
  - [Flixster](http://socialcomputing.asu.edu/datasets/Flixster): Flixster is a social movie site allowing users to share movie ratings, discover new movies and meet others with similar movie taste.
  - [IMDB](http://komarix.org/ac/ds/): This is a link dataset built with permission from the Internet Movie Data (IMDB).
  

### Trust
  - [CiaoDVD & Epinions](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm): The CiaoDVD is a dataset crawled from the entire category of DVDs, and the Epinions dataset for each user, in their profile, it contains their ratings and trust relations. For each rating, the product name and its category, the rating score, the time point when the rating is created, and the helpfulness of this rating.
  
  
### Anime
  - [Anime Recommendations Database](https://www.kaggle.com/CooperUnion/anime-recommendations-database): This data set contains information on user preference data from 73,516 users on 12,294 anime. Each user is able to add anime to their completed list and give it a rating and this data set is a compilation of those ratings.
  - [Anime Data](https://www.kaggle.com/canggih/anime-data-score-staff-synopsis-and-genre): Japanese animation, which is known as anime, has become internationally widespread nowadays. This dataset provides data on anime taken from Anime News Network.
  
  
### Food
  - [Resturant and Constumer](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data): This dataset was obtained from a recommender system prototype, with the task to generate a top-n list of restaurants according to the consumer preferences.
  - [Chicago Entree](http://archive.ics.uci.edu/ml/datasets/Entree+Chicago+Recommendation+Data): This is a dataset containing a record of user interactions with the Entree Chicago restaurant recommendation system.
 
  
### Games
  - [Steam Video Games](https://www.kaggle.com/tamber/steam-video-games/data): This dataset is a list of user behaviors, with columns such as user-id, game-title, behavior-name, value. The behaviors included are 'purchase' and 'play'. The value indicates the degree to which the behavior was performed - in the case of 'purchase' the value is always 1, and in the case of 'play' the value represents the number of hours the user has played the game.
  - [Steam Reviews Dataset](https://www.kaggle.com/luthfim/steam-reviews-dataset): This dataset contains reviews from Steam's best selling games as February 2019.
  
 
### Jokes
  - [Jester](http://www.ieor.berkeley.edu/~goldberg/jester-data/): This is a Joke dataset containing 4.1 million continuous ratings (-10.00 to +10.00) of 100 jokes from 73,496 users.
  
  
### Other 
  - [Citation Network](https://www.aminer.cn/citation#b541): The data set is designed for research purpose only. The citation data is extracted from DBLP, ACM, MAG (Microsoft Academic Graph), and other sources. The first version contains 629,814 papers and 632,752 citations. Each paper is associated with abstract, authors, year, venue, and title.
  - [YAGO](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/): YAGO is a huge semantic knowledge base, derived from Wikipedia WordNet and GeoNames. Currently, YAGO has knowledge of more than 10 million entities (like persons, organizations, cities, etc.) and contains more than 120 million facts about these entities.
  - [Complete Collection of Kaggle Datasets](https://www.kaggle.com/stefanoleone992/complete-collection-of-kaggle-datasets): (below is more information pertaining to this dataset)
  > - Context: For many data analysts it is often complicated to find the right dataset for a project or to make some practice, so this collection of Kaggle datasets helps them to explore the available opportunities that Kaggle offers.
  
  > - Content: Part of the data has been first collected using the Kaggle API to retrieve the full list datasets, then each URL reference has been leveraged with a Python script in order to retrieve more detailed information.
  
 
 
# A collection of resources for Recommender Systems (RecSys)

## Recommendation Algorithms

- Recommender Systems Basics
  - [Wikipedia](https://en.wikipedia.org/wiki/Recommender_system)
- Nearest Neighbor Search
  - [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
  - [sklearn.neighbors](http://scikit-learn.org/stable/modules/neighbors.html)
  - [Benchmarks of approximate nearest neighbor libraries](https://github.com/erikbern/ann-benchmarks)
- Classic Matrix Facotirzation
  - [Matrix Factorization: A Simple Tutorial and Implementation in Python](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/)
  - [Matrix Factorization Techiques for Recommendaion Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- Singular Value Decomposition (SVD)
  - [Wikipedia](https://en.wikipedia.org/wiki/Singular-value_decomposition)
- SVD++
  - [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)
- Content-based CF / Context-aware CF
  - there are so many ...
- Advanced Matrix Factorization
  - [Probabilistic Matrix Factorization](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)
  - [Fast Matrix Factorization for Online Recommendation with Implicit Feedback](https://dl.acm.org/citation.cfm?id=2911489)
  - [Collaborative Filtering for Implicit Feedback Datasets](http://ieeexplore.ieee.org/document/4781121/)
  - [Factorization Meets the Item Embedding: Regularizing Matrix Factorization with Item Co-occurrence](https://dl.acm.org/citation.cfm?id=2959182)
- Factorization Machine
  - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  - [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134)
- Sparse LInear Method (SLIM)
  - [SLIM: Sparse Linear Methods for Top-N Recommender Systems](http://glaros.dtc.umn.edu/gkhome/node/774)
  - [Global and Local SLIM](http://glaros.dtc.umn.edu/gkhome/node/1192)
- Learning to Rank
  - [Wikipedia](https://en.wikipedia.org/wiki/Learning_to_rank)
  - [BPR: Bayesian personalized ranking from implicit feedback](https://dl.acm.org/citation.cfm?id=1795167)
  - [WSABIE: Scaling Up To Large Vocabulary Image Annotation](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)
  - [Top-1 Feedback](http://proceedings.mlr.press/v38/chaudhuri15.pdf)
  - [k-order statistic loss](http://www.ee.columbia.edu/~ronw/pubs/recsys2013-kaos.pdf)
  - [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=3015834)
  - [The LambdaLoss Framework for Ranking Metric Optimization](https://dl.acm.org/citation.cfm?id=3271784)
- Cold-start
  - [Deep content-based music recommendation](https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation)
  - [DropoutNet: Addressing Cold Start in Recommender Systems](https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems)
- Network Embedding
  - [awesome-network-embedding](https://github.com/chihming/awesome-network-embedding)
  - [Item2vec](https://arxiv.org/abs/1603.04259)
  - [entity2rec](https://dl.acm.org/citation.cfm?id=3109889)
- Sequential-based
  - [Factorizing Personalized Markov Chains for Next-Basket Recommendation](https://dl.acm.org/citation.cfm?id=1772773)
  - [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)
- Translation Embedding
  - [Translation-based Recommendation](https://dl.acm.org/citation.cfm?id=3109882)
  - [Translation-based Factorization Machines for Sequential Recommendation](https://dl.acm.org/citation.cfm?id=3240356)
- Graph-Convolution-based
  - [GraphSAGE: Inductive Representation Learning on Large Graphs](https://dl.acm.org/doi/10.5555/3294771.3294869)
  - [PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973)
- Knowledge-Graph-based
  - [Collaborative knowledge base embedding for recommender systems](https://dl.acm.org/doi/10.1145/2939672.2939673)
  - [Knowledge Graph Convolutional Networks for Recommender Systems](https://dl.acm.org/citation.cfm?id=3313417)
  - [KGAT: Knowledge Graph Attention Network for Recommendation](https://dl.acm.org/authorize.cfm?key=N688414)
  - [Ripplenet: Propagating user preferences on the knowledge graph for recommender systems](https://dl.acm.org/doi/10.1145/3269206.3271739)
- Deep Learning
  - [Deep Neural Networks for YouTube Recommendations](https://ai.google/research/pubs/pub45530)
  - [Deep Learning based Recommender System: A Survey and New Perspectives](https://arxiv.org/abs/1707.07435)
  - [Neural Collaborative Filtering](https://dl.acm.org/citation.cfm?id=3052569)
  - [Collaborative Deep Learning for Recommender Systems](http://www.wanghao.in/CDL.htm)
  - [Collaborative Denoising Auto-Encoders for Top-N Recommender Systems](https://dl.acm.org/citation.cfm?id=2835837)
  - [Collaborative recurrent autoencoder: recommend while learning to fill in the blanks](https://dl.acm.org/citation.cfm?id=3157143)
  - [TensorFlow Wide & Deep Learning](https://www.tensorflow.org/tutorials/wide_and_deep)
  - [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/pub45530.html)
  - [Collaborative Memory Network for Recommendation Systems](https://arxiv.org/abs/1804.10862)
  - [Variational Autoencoders for Collaborative Filtering](https://dl.acm.org/citation.cfm?id=3186150)
  
## Online Courses
- [Recommender Systems Specialization](https://zh-tw.coursera.org/specializations/recommender-systems), University of Minnesota
- [Introduction to Recommender Systems: Non-Personalized and Content-Based](https://zh-tw.coursera.org/learn/recommender-systems-introduction), University of Minnesota

## RecSys-related Competitions
- [Kaggle](https://www.kaggle.com/) - product recommendations, hotel recommendations, job recommendations, etc.
- ACM RecSys Challenge
- [WSDM Cup 2018](https://wsdm-cup-2018.kkbox.events/)
- [Million Song Dataset Challenge](https://www.kaggle.com/c/msdchallenge)
- [Netflix Prize](https://www.netflixprize.com/)

## Tutorials
- RecSys tutorials
  - [2014](https://recsys.acm.org/recsys14/tutorials/)
  - [2015](https://recsys.acm.org/recsys15/tutorials/)
  - [2016](https://recsys.acm.org/recsys16/tutorials/)
  - [2017](https://recsys.acm.org/recsys17/tutorials/)
  - [2018](https://recsys.acm.org/recsys18/tutorials/)
- [Kdd 2014 Tutorial - the recommender problem revisited](https://www.slideshare.net/xamat/kdd-2014-tutorial-the-recommender-problem-revisited)

## Articles
- [Matrix Factorization: A Simple Tutorial and Implementation in Python](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/)

## Conferences
- [RecSys – ACM Recommender Systems](https://recsys.acm.org/)
