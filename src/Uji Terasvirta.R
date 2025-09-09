library(readxl)
library(moments)
library(TSA)
file_path <- "C:\\Users\\Lenovo\\Downloads\\MATKUL\\BISMILLAH S.Si\\4. Kolokium\\Data\\Data Emas.xlsx"
df <- read_excel(file_path, sheet = 1)
#Uji Terasvirta
terasvirta.test <- function(x) {
  residuals <- lm(x ~ lag(x, 1))$residuals  #model linear
  n <- length(residuals)
  Y <- residuals[4:n]
  Z <- residuals[3:(n-1)]
  Z2 <- Z^2
  Z3 <- Z^3
  model <- lm(Y ~ Z + Z2 + Z3)
  summary(model)}
terasvirta.test(df$Terakhir)