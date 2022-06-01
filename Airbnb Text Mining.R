
##Airbnb Text Mining Codes##

#load packages. If the packages don't load, run the above set of lines first. 
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggmap)
library(cluster)   
library(tm)
library(topicmodels)
library(slam)
library(SnowballC)
library(DT)

#combine reviews data and listings data
text.mining <- read_csv("TextMiningData.csv")

#change the filtered data name
reviews.filtered <- text.mining

#Creating DTM and getting the Word Count
#changing the column names so that the alogorithm detects them 
reviews.filtered <- reviews.filtered %>% mutate(doc_id=reviewer_id,text=comments)
#creating a database for texts
review.corpus <- VCorpus(DataframeSource(reviews.filtered))

#cleaning the text data
#Mac Users, use the following two lines
review.corpus.clean <- tm_map(review.corpus, content_transformer(function(x) iconv(x, to='UTF-8-MAC', sub='byte')))
review.corpus.clean <- tm_map(review.corpus.clean, content_transformer(tolower)) #Interface to apply transformation functions to corpora.
#All users, run these lines
review.corpus.clean <- tm_map(review.corpus.clean, removeWords, stopwords("english"))
review.corpus.clean <- tm_map(review.corpus.clean, removePunctuation)
review.corpus.clean <- tm_map(review.corpus.clean, removeNumbers)
review.corpus.clean <- tm_map(review.corpus.clean, stemDocument, language="english") #perform stemming which truncates words
review.corpus.clean <- tm_map(review.corpus.clean,stripWhitespace)

dtm <- DocumentTermMatrix(review.corpus.clean)

#checking the first ten documents using the words: clean and room
inspect(dtm[1:10,c("nyc","room")])

## Check frequency and make frequency plot
freq <- colSums(as.matrix(dtm)) 

###Very hard to see, so let's make a visualization###
#Step 1: make a separate term.count datasheet
term.count <- as.data.frame(as.table(dtm)) %>%
  group_by(Terms) %>%
  summarize(n=sum(Freq))

#Step 2:Keep High Frequency words only, e.g, 0.25%
term.count %>% 
  filter(cume_dist(n) > 0.9975) %>% #cume_dist is the cumulative distribution function which gives the proportion of values less than or equal to the current rank
  ggplot(aes(x=reorder(Terms,n),y=n)) + geom_bar(stat='identity') + 
  coord_flip() + xlab('Counts') + ylab('')

#Another way to find the frequent terms 
findFreqTerms(dtm, lowfreq=150)

# find terms correlated with "great" 
great <- data.frame(findAssocs(dtm, "great", 0.09))
great %>%
  add_rownames() %>%
  ggplot(aes(x=reorder(rowname,great),y=great)) + geom_point(size=4) + 
  coord_flip() + ylab('Correlation') + xlab('Term') + 
  ggtitle('Terms correlated with Great') + theme(text=element_text(size=20))

# find terms correlated with "stay"
stay <- data.frame(findAssocs(dtm, "stay", 0.15))
stay %>%
  add_rownames() %>%
  ggplot(aes(x=reorder(rowname,stay),y=stay)) + geom_point(size=4) + 
  coord_flip() + ylab('Correlation') + xlab('Term') + 
  ggtitle('Terms correlated with Stay') + theme(text=element_text(size=20))

# find terms correlated with "host" 
host <- data.frame(findAssocs(dtm, "host", 0.1))
host %>%
  add_rownames() %>%
  ggplot(aes(x=reorder(rowname,host),y=host)) + geom_point(size=4) + 
  coord_flip() + ylab('Correlation') + xlab('Term') + 
  ggtitle('Terms correlated with Host') + theme(text=element_text(size=20))

 ## Make a wordcloud
#install.packages("wordcloud")
library(wordcloud)
popular.terms <- filter(term.count,n > 500)
wordcloud(popular.terms$Terms,popular.terms$n,colors=brewer.pal(8,"Dark2"))


###########################################################################################
# SENTIMENT ANALYSIS
###########################################################################################
library(SentimentAnalysis)

#if the above line does not work and you have a MAC, try this
recode <-function(x) {iconv(x, to='UTF-8-MAC', sub='byte')}
sentiment <- analyzeSentiment(recode(reviews.filtered$text))

#all users, run the line below
sent_df = data.frame(polarity=sentiment$SentimentQDAP, business = reviews.filtered, stringsAsFactors=FALSE)

# Plot results and check the correlation betweeen polarity and review stars #
#First, check the summary statistics of the polarity score
summary(sent_df$polarity)

#Second, check for the missing value. Why is there a missing value? 
sent_df %>% filter(is.na(polarity)==TRUE)%>%select(business.text)

#Now, correct for that missing value, NA
sent_df$polarity[is.na(sent_df$polarity)]=0
sent_df$business.review_scores_rating<-as.numeric(sent_df$business.review_scores_rating)

#check the correlation between star ratings and polarity score
cor(sent_df$polarity,sent_df$business.review_scores_rating)

#visualize the correlation 
sent_df %>%
  group_by(business.review_scores_rating) %>%
  summarize(mean.polarity=mean(polarity,na.rm=TRUE)) %>%
  ggplot(aes(x=business.review_scores_rating,y=mean.polarity)) +  geom_bar(stat='identity',fill="blue") +  
  ylab('Mean Polarity') + xlab('Review Score')  + theme(text=element_text(size=20))

#visualize using a line chart
sent_df %>% ggplot(aes(x=business.review_scores_rating, y=polarity)) +
  geom_point() +
  geom_smooth(method=lm) + 
  ggtitle("Polarity Score by Review Rating")+xlab("Review Rating")+ ylab("Polarity Score")


###########################################################################################
# TOPIC MODELING
# R package: "topicmodels"
###########################################################################################

## set.up.dtm.for.lda.1
library(topicmodels)
library(slam)

dtm.lda <- removeSparseTerms(dtm, 0.98)
review.id <- reviews.filtered$review_id[row_sums(dtm.lda) > 0]
dtm.lda <- dtm.lda[row_sums(dtm.lda) > 0,]

## run LDA algorithm - WARNING: takes a while to run!k = number of topics 
lda.airbnb <- LDA(dtm.lda,k=20,method="Gibbs",
                control = list(seed = 2011, burnin = 1000,
                               thin = 100, iter = 5000))
#to be safe, save the file 
save(lda.airbnb,file='lda_results.rda')

## load results (so you don't have to run the algorithm)
load('lda_results.rda')

#get the posterior probability of the topics for each document and of the terms for each topic
post.lda.airbnb <- posterior(lda.airbnb) 

##  sum.lda to get a matrix with topic by terms "cleaning data process"
sum.terms <- as.data.frame(post.lda.airbnb$terms) %>% #matrix topic * terms
  mutate(topic=1:20) %>% #add a column
  gather(term,p,-topic) %>% #gather makes wide table longer, key=term, value=p, columns=-topic (exclude the topic column)
  group_by(topic) %>%
  mutate(rnk=dense_rank(-p)) %>% #add a column
  filter(rnk <= 10) %>%
  arrange(topic,desc(p)) 

## see the words in each topic
# words in topic 1 - Traveling Intricacies
sum.terms %>%
  filter(topic==1) %>%
  ggplot(aes(x=reorder(term,p),y=p)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Term')+ylab('Probability')+ggtitle('Topic 1') + theme(text=element_text(size=20))

#words in topic 2 - Amenities and Preferences
sum.terms %>%
  filter(topic==2) %>%
  ggplot(aes(x=reorder(term,p),y=p)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Term')+ylab('Probability')+ggtitle('Topic 2') + theme(text=element_text(size=20))

#words in topic 3 - Communication and Access
sum.terms %>%
  filter(topic==3) %>%
  ggplot(aes(x=reorder(term,p),y=p)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Term')+ylab('Probability')+ggtitle('Topic 3') + theme(text=element_text(size=20))

#words in topic 4 - Visual Experiences
sum.terms %>%
  filter(topic== 4) %>%
  ggplot(aes(x=reorder(term,p),y=p)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Term')+ylab('Probability')+ggtitle('Topic 4') + theme(text=element_text(size=20))

#words in topic 5 - Host
sum.terms %>%
  filter(topic== 5) %>%
  ggplot(aes(x=reorder(term,p),y=p)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Term')+ylab('Probability')+ggtitle('Topic 5') + theme(text=element_text(size=20))

#words in topic 7 - Recommendation Based
sum.terms %>%
  filter(topic== 7) %>%
  ggplot(aes(x=reorder(term,p),y=p)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Term')+ylab('Probability')+ggtitle('Topic 7') + theme(text=element_text(size=20))

#words in topic 10 - Location
sum.terms %>%
  filter(topic==10) %>%
  ggplot(aes(x=reorder(term,p),y=p)) + geom_bar(stat='identity') + coord_flip() + 
  xlab('Term')+ylab('Probability')+ggtitle('Topic 10') + theme(text=element_text(size=20))


