file.df<- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", sep=",", header = FALSE)

library('caret')
install.packages("MASS")
install.packages("gains")
library(MASS)
library(gains)

#File spambase.names read
data = read.delim("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names", sep=" ", header=FALSE)
names<-as.factor(data[-(1:30),1])
colnames(file.df) = names
colnames(file.df)[58]<- "Type.Spam"

#Normalising the data
file.n.df<-(sapply(file.df[,-58],scale))
file.n.df<-cbind(file.n.df,file.df$Type.Spam)
colnames(file.n.df)[58]<- "Type.Spam"
file.n.df<-data.frame(file.n.df)

#Calculating the difference between the average value of spam and non spam values
difference<-abs(diff(as.matrix(aggregate(.~Type.Spam,file.n.df, mean))))
#Top 10 predictors
use_predictors<-head(sort(difference[,-1],decreasing=TRUE),10)

#Only 10 predictors and Type.Spam
perform.df<-file.n.df[,c("word_freq_your.","word_freq_000.","word_freq_remove.","char_freq_...4","word_freq_you.","word_freq_free.","word_freq_business.","word_freq_hp.","capital_run_length_total.","word_freq_our.","Type.Spam")]

#Q2
#Partition Dataset
set.seed(13)
train.index <- createDataPartition(perform.df$Type.Spam, p = 0.8, list = FALSE)
train.df <- perform.df[train.index, ]
valid.df <- perform.df[-train.index, ]

#LDA
LDA_ans<-lda(Type.Spam ~., data = train.df)

#Evaluating the model

# predict
pred <- predict(LDA_ans,valid.df)
names(pred)  ## "class", "posterior", "x"

# check model accuracy - Confusion Matrix
table(pred$class, valid.df$Type.Spam)  #predicted vs actual

gain <- gains(valid.df$Type.Spam, pred$x)

### Plot Lift Chart
plot(c(0,gain$cume.pct.of.total*sum(valid.df$Type.Spam))~c(0,gain$cume.obs), 
     xlab = "# email messages", ylab = "Cumulative", main = "", type = "l")
lines(c(0,sum(valid.df$Type.Spam))~c(0, dim(valid.df)[1]), lty = 5)

### Plot decile-wise chart
heights <- gain$mean.resp/mean(valid.df$Type.Spam)
decile_lift <- barplot(heights, names.arg = gain$depth,  ylim = c(0,3), col = "red",  
                     xlab = "Percentile", ylab = "Mean Response-Type Spam", 
                     main = "Decile-wise lift chart")

#Class of Interest : SPAM or 1

#Q3 Using the confusion matrix, lift chart, and decile chart for the validation dataset
#evaluate the effectiveness of the model in identifying spams.

#CONFUSION MATRIX (line 46 above)
Accuracy <- ((531+231)/920)*100 #82.82609%
Sensitivity<- (231/(231+125))*100 #sensitivity <- (Actual spam identified as spam-Correctly classified/ Actual spam)*100
#The model is 64.88% accurate in correctly classifying spam from actual spams

#LIFT CHART 
#After examining 200 cases, 175 have been correctly identified as Spam whereas only 70 would have been identified as spam if choosen randomly
#The dotted line indicating the result of identifying Spam when no model is used and the curved line indicating the model performance. By comparing both,
#we can conclude that our model will gain the most when 400 cases are used, approx 300 would be identified as spam ( as the greater the area between the lift curve and the baseline, the better it is)

#DECILE LIFT CHARTS
#By comparing the model developed to "no model/pick randomly/naive approach/no model baseline" we measure the ability of our model and interpret that our model is 
#doing 2.3 times approx better job in identifying important class(Spam in our case) than a normal random selection of 10 percentile would have done. In other words, it can identify 2.3 times more spam transactions 
#than would a random selection would have identified when 10% of the email email messages are considered. In most probable i.e top decile, model is slightly more than twice likely to identify the important class as compared to average prevelance
#Second decile is slightly more that first decile indicating that there was scope for our model to slightly improve more.
#Further decreasing decile indicate that our model is going a great job. Second decile would help us identify 2.45 more spams than a naive selection would.
