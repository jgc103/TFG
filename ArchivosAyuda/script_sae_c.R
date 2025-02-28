library(keras)
library(caret)


setwd("~/Docencia/Curs_2020_2021_2S/UB/CursDLOmics/SAE_SNPsGeneExpression")



##############################

load("htseq.imp.RData")
T_seq<-matriu.expressio.imp$ximp
dim(T_seq)

load("Tensor_SNP.RData")
dim(T_SNP)   # SNP 3 niveles codificados por (0-1-2); one-hot encoding de (112,2956) a (112,8868)



n.obs<-nrow(T_SNP)
snp_shape <- ncol(T_SNP)
htseq_shape <- ncol(T_seq)


### Adding binomial noise 
T_noise<-matrix(rbinom(n.obs*snp_shape,1,0.01),n.obs,snp_shape)
ind = which(T_noise < T_SNP)  # 0 en noise i 1 en SNP
T_noise_SNP<-T_noise
T_noise_SNP[ind]<-T_SNP[ind]

# Tsnp = 100 010 ; Tnoise =010 100
# ind = true false false false true false
# TnoiseSNP = 010 100
# TnoiseSNP = 110 110


############################################### AE1

# input layer
entrada_SNP <-layer_input(shape = c(NULL, snp_shape),
              dtype = 'float32')


output_enc1<-
  entrada_SNP %>% 
  layer_dense(
    name = "encoder1_1",
    units = 1000,
    kernel_regularizer = regularizer_l2(2e-04)  # regularizador
  ) %>%
  layer_dense(
    name = "encoder1_2",
    units = 300,
    activation = "relu",
    kernel_regularizer = regularizer_l2(2e-04) 
  ) 

# opcion no simetrica 
output_dec1<-output_enc1 %>% 
  layer_dense(units=snp_shape,
              activation="sigmoid",
              kernel_regularizer = regularizer_l2(2e-04))

aen1<-keras_model(inputs =c(entrada_SNP),outputs = output_dec1)

summary(aen1)


aen1 %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy")


aen1 %>% fit(
  x=list(as.matrix(T_noise_SNP)),
  y=list(as.matrix(T_SNP)),
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
)



################################################## AE2



output_enc2<-output_enc1 %>% 
  layer_dense(
    name = "encoder2_1",
    units = 100,
    kernel_regularizer = regularizer_l2(2e-04)
  ) %>%
  layer_dense(
    name = "encoder2_2",
    units = 50,
    activation = "relu",
    kernel_regularizer = regularizer_l2(2e-04) 
  ) 



output_dec2<-output_enc2 %>% 
  layer_dense(units=snp_shape,
              activation="sigmoid",
              kernel_regularizer = regularizer_l2(2e-04))



sae <- keras_model(inputs =c(entrada_SNP), outputs = output_dec2)
summary(sae)

aen1 %>% freeze_weights() 

summary(sae)

aen1 %>% compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy")

sae %>% compile(optimizer = "rmsprop",
                loss = "binary_crossentropy")




sae %>% fit(
  x=list(as.matrix(T_noise_SNP)),
  y=list(as.matrix(T_SNP)),
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
)



################################# Fine tuning


aen1 %>% unfreeze_weights(from = "encoder1_2" )
summary(sae)

aen1 %>% compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy")

sae %>% compile(optimizer = "rmsprop",
                loss = "binary_crossentropy")

sae %>% fit(
  x=list(as.matrix(T_noise_SNP)),
  y=list(as.matrix(T_SNP)),
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
)



################################## MLP + SAE
mlpsae <- output_enc2 %>%
  layer_dense(units = 10,
              activation = "relu",
              name = "interna_mlpsae1",
              kernel_regularizer = regularizer_l2(2e-04)) %>%
  layer_dense(units = htseq_shape,
              activation = "linear",
              name = "out_mlpsae")


mlpsae <-keras_model(inputs =c(entrada_SNP),outputs = mlpsae)
summary(mlpsae)



sae %>% freeze_weights()
summary(mlpsae)


mlpsae %>% compile(optimizer = "rmsprop",
                         loss = "mse",
                         metrics = "mse")


mlpsae %>% fit(
  x = list(as.matrix(T_noise_SNP)),
  y = as.matrix(T_seq), 
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
)



############################### Fine tuning 
summary(mlpsae)
unfreeze_weights(sae, from = "encoder2_2")
summary(mlpsae)


mlpsae %>% compile(optimizer = "rmsprop", loss = "mse", metrics = "mse")
sae %>% compile(optimizer =  "rmsprop", loss = "mse", metrics = "mse")


mlpsae %>% fit(
  x = list(as.matrix(T_noise_SNP)),
  y = as.matrix(T_seq), 
  epochs = 25,
  batch_size=64,
  validation_split = 0.2
)


yhat<-predict(mlpsae,as.matrix(T_noise_SNP))  #


dim(yhat)
ix<-400
cor(T_seq[,ix],yhat[,ix])
plot(T_seq[,ix],yhat[,ix])
abline(lm(yhat[,ix]~T_seq[,ix]))

vcor<-diag(cor(T_seq,yhat))
summary(vcor)
boxplot(vcor)
hist(vcor)


