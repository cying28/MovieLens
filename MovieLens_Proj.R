rm(list = ls())
#library(ggplot2)
################################
# Generate Dataset, Training & Evaluation
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")

# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


######################################################################
#I. Data Summary #########################################

#1) View categories of training set and check dimensions of training and validation set
head(edx)

tr<-dim(edx)
val<-dim(validation)
print(c(tr,val)) #Dimensions of training and validation

val_p<-val[1]*100/(tr[1]+val[1])
print(val_p) #True Validation Percentage

#2) Check if there's NA values
sapply(edx, function(x) sum(is.na(x)))

#3) Identify distinct users and make sure one movie matches one movieID
edx %>%
  summarize(n_users = n_distinct(userId),
            n_movieId = n_distinct(movieId),
            n_movies = n_distinct(title))

#4) Extract more useful information (extracting year from title)
edx_1 <- mutate(edx, year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$"))
validation_1 <- mutate(validation, year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$"))


#5) View ratings count and distribution
ratings_count<-
  edx %>% 
  group_by(rating) %>% 
  summarise(count=n()) %>% 
  arrange(desc(count)) 
print(ratings_count)

#6) Mean and Quantiles for number of ratings per user ID
user<- edx %>% count(userId)
mean(user$n)
median(user$n)
max(user$n)
quantile(user$n)

#7) Genre count ranked from high to low
genre_count<-
  edx %>% 
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
print(genre_count)

#II. Data Visualization #########################################
#1) Genre Distribution
re_gen<- factor(genre_count$genres, as.character(genre_count$genres))
gen_dist<- ggplot(genre_count, aes(y=count))
gen_dist +
  geom_bar(aes(x=re_gen), data=genre_count, stat="identity") + 
  coord_flip() +
  ggtitle("Genre Distribution")

#2) Ratings Distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "grey") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  ggtitle("Ratings Distribution")

#3) Movie/year Distribution (data until 2008)
yearly_movie<-
  edx_1 %>%
  select(movieId, year) %>%
  group_by(year) %>% 
  summarise(count = n())  %>%
  arrange(year)

yearly_movie %>%
  ggplot(aes(x = year, y = count)) +
  geom_line(color="black") +
  ggtitle("Movie/year Distribution")

yearly_movie[yearly_movie$count==max(yearly_movie$count), ]

#4) Scatter plot of unique users and number of ratings
edx %>% 
  count(userId) %>%
  ggplot(aes(x=userId,y=n)) +
  geom_point() +
  xlab("User ID") +
  ylab("Number of Ratings") +
  ggtitle("Unique Users and Number of Ratings") 

#5) Hexbin plot of average movie rating and number of ratings
avg_rate<- edx %>% group_by(movieId) %>% summarize(avg=mean(rating))
num_rate<- edx %>% count(movieId) 
left_join(avg_rate,num_rate) %>% ggplot(aes(x=n,y=avg))+
  geom_bin2d()+
  theme_bw()+
  xlab("Number of Ratings") +
  ylab("Average Rating") +
  ggtitle("Hexbin avg rating and # of ratings") 

#III. Algorithm
#Reference: https://rafalab.github.io/dsbook/

#Loss Function (Evaluation Formula)
RMSE <- function(tru_ratings, pred_ratings){
  sqrt(mean((tru_ratings - pred_ratings)^2))
}

#1) Starting RMSE 
mu_hat <- mean(edx$rating)
RMSE_1<-RMSE(validation$rating, mu_hat) #1.061
rmse_table<- data.table(Method="Naive Mean", RMSE=RMSE_1)
rmse_table %>%knitr::kable()

#2) Modeling using movieId
movie_avg <- edx %>%
  group_by(movieId) %>%
  summarize(b_movie = mean(rating - mu_hat))

predict_2 <- validation %>%
  left_join(movie_avg, by='movieId') %>%
  pull(b_movie) + mu_hat

RMSE_2 <- RMSE(predict_2, validation$rating) #0.9439
rmse_table <- bind_rows(rmse_table, 
                          data_frame(Method="Movie Model",
                                     RMSE = RMSE_2))
rmse_table %>%knitr::kable()

#3) Modeling using userId + movieId
user_avg<- edx %>%
  left_join(movie_avg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_user=mean(rating-mu_hat-b_movie))
  
predict_3<- validation %>%
  left_join(movie_avg, by="movieId") %>%
  left_join(user_avg, by="userId") %>%
  mutate(pred3=mu_hat+b_movie+b_user) %>%
  pull(pred3)

RMSE_3 <- RMSE(predict_3, validation$rating) #0.86535
rmse_table <- bind_rows(rmse_table, 
                        data_frame(Method="Movie+User Model",
                                   RMSE = RMSE_3))
rmse_table %>%knitr::kable()

#4) Modeling using userId + movieId + movie year

year_avg<- edx_1 %>%
  left_join(movie_avg, by="movieId") %>%
  left_join(user_avg, by="userId") %>%
  group_by(year) %>%
  summarize(b_year=mean(rating-mu_hat-b_movie-b_user))

predict_4<- validation_1 %>%
  left_join(movie_avg, by="movieId") %>%
  left_join(user_avg, by="userId") %>%
  left_join(year_avg, by="year") %>%
  mutate(pred4=mu_hat+b_movie+b_user+b_year) %>%
  pull(pred4)

RMSE_4 <- RMSE(predict_4, validation_1$rating) #0.86500
rmse_table <- bind_rows(rmse_table, 
                        data_frame(Method="Movie+User+Year Model",
                                   RMSE = RMSE_4))
rmse_table %>%knitr::kable()

#5) Regularized movie + user + year
reg_mov_avg<-edx_1 %>%
  group_by(movieId) %>%
  summarize(b_movie_reg=sum(rating-mu_hat)/(n()+4), n_movie=n()) #lambda=4
  
predict_5<- validation_1 %>%
  left_join(reg_mov_avg, by="movieId") %>%
  left_join(user_avg, by="userId") %>%
  left_join(year_avg, by="year") %>%
  mutate(pred_5=mu_hat+b_movie_reg+b_user+b_year) %>%
  pull(pred_5)

RMSE_5 <- RMSE(predict_5, validation_1$rating) #0.86488
rmse_table <- bind_rows(rmse_table, 
                        data_frame(Method="Reg_Movie+User+Year Model",
                                   RMSE = RMSE_5))
rmse_table %>%knitr::kable()

