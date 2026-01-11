# Attention is all you need (Transformer)

## Recurent neural network

used in squence, and NLP, we devide the sencestnce into little chuncks for words called squence the we pass each sequnce to one RNN which took
the squence and intial state/ privous state as a input and then output a single word
./images/img1.*

we are porcessing one word at a time in on RRN, and also it takes in the privous state it made the compution slow, it also had vanishing
and exploding gredeint problem , and we pass only prvious state it can not maintain the realtion betwenn suppose squence num 1 and squence num 10

## Transformer
./images/img2.*

to slove the problems in RNN we use transformer, transformer has 2 part encoder and decoder.
### Encoder

./images/img3.*

Encoder start with input embedding, suppose we have a sentance, we 1st split them into sinlge words or more smaller parts this is called tokinzation
then we map this words to a number i.e postion in our voculabary(all the words laters a model knows), after coverting the token to the number we
map it to a vector of size 512 we alwayse map same word with same vector, this is called word embedding however during traiing the embedding might change
but it will alwayse map to the origanl word

### Postional Encoding
./images/img4
we want each word to carry some infomation about the postion in the sentance, we want model to treat word that apper close to each other close
and disnt to each other as disnt, this are only computed onces and reused for every sentence during training and infernce,
Postional encoding for even is places in postion vector in calcauted like this PE(pos,2i) = \sin \frac{pos}{10000^{\frac{2i}{d_{model}}}} and for
odd postion is calulted by PE(pos,2i) = \cos \frac{pos}{10000^{\frac{2i}{d_{model}}}} and for

