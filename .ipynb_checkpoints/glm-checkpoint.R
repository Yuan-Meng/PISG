setwd("/Users/apple/Desktop/My folders/Research/projects/inProgress/PISG/cogsci20/writeup")
df <- read.csv("check.csv")


library(lme4)


mod1 <- glmer(buy ~ checked + (1|participant) + (1|trial), family=binomial, data=df, REML=FALSE)
mod2 <- glmer(buy ~ 1+ (1|participant) + (1|trial), family=binomial, data=df, REML=FALSE)
anova(mod1, mod2)

