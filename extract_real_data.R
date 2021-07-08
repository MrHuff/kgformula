setwd("~/Documents/phd_projects/kgformula")
library(qte)
data(lalonde)
dat<-lalonde.exp

write.csv(dat,"lalonde.csv", row.names = TRUE)
