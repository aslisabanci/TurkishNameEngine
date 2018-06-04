# TurkishNameEngine

Inspired by the mini "Dinosaur name generator" project I did a while ago, I wanted to re-use the core code (with some modifications and refactorings of my own) and try to generate Turkish first names. The project was about building a simple recurrent neural network using numpy and it was a part of the Sequence Models course of Andrew Ng's awesome Deep Learning Specialization series on Coursera. 


## Main Flow

The project builds a simple character level language model to generate new first names. Roughly, the steps are to:  

- Initialize the weights
- Forward propagate and calculate the loss
- Backward propagate and calculate the gradients, using the loss
- Clip the gradients with a max value, so that they don't explode
- Update the weights using the gradients

The input we provide is the list of characters of each name in the dataset; and the output at every time step is the next character to be used. At each time step, our recurrent neural network is picking a character, given the previous character, according to a probability distribution.


## Preprocessing the data 
I wanted to try this model with two approaches: 
- Get the most common 10 thousand names within the whole dataset
- Get 3 million names, regardless of their frequency

For both approaches, I needed to clean the data first because some of the names contained characters like dot, paranthese, hyphen, number and what not. 

Also there were some non-Turkish names in the dataset, so I removed names that contain "x", "w", "q" as these letters are not in the Turkish alphabet. 

Lastly, there were a lot of first names consisting of two or three names. Since I didn't want my model to generate names consisting of two or more names, I split these into single ones. While procesing the names one by one, I added a split name into my list of names, if it hasn't been added in the earlier iterations. 

There were a lot of names with this style within my 10K and 3m dataset. So after reducing them to single names, I only ended up with ~700 and ~5000 distinct names respectively.

## Getting predictions
I wanted to see how familiar the names will look, when trained with the most common Turkish names (10K dataset) and when trained with most of the names coming from the longer tail (3m dataset).

So I took 10 different output samples from each model.

| 10K dataset | 3m dataset |
|---|---|
| Sin, Badir, Sedef, Akmul, Salan, Betad, Behan, Siga, Kanefi, Sac | Bevit, Ezer, Kumul, Zarit, Semiye, Gursap, Hukdeye, Sevar, Tahiye, Ozar
| Vuran, Bahir, Urdi, Akir, Sam, Curec, Celkemi, Sula, Kari, Saba | Goley, Niyur, Menan, Apmat, Mindorlah, Azda, Sulki, Guka, Kime, Ergi
| Radugi, Dukdur, Maha, Zusuf, Mayih, Nefah, Aysu, Selsem, Gabsun, Kahat | Zuraye, Rettime, Sengelli, Gulcime, Sortedi, Sugon, Zaki, Memra, Kunni, Hucan
| Bercan, Habbat, Nultan, Ayiz, Yurev, Havgi, Bakes, Yaydil, Melif, Kalis | Gungu, Aybeden, Yavrun, Samine, Ruydar, Arnur, Hazti, Vuteton, Cahan, Talize
| Baytul, Sicen, Ekep, Ale, Murgal, Kerif, Furem, Zihan, Kubza, Luse | Tekayu, Cabilatviye, Feyhan, Badabed, Sasir, Gugiki, Guhteviz, Sehhun, Sakser, Edana 
| Becer, Hase, Gahat, Betri, Sidar, Erzan, Gulsul, Benan, Kevay, Efere | Uzatil, Iren, Buhneyan, Sefdan, Vezhap, Fimnan, Bikan, Hirdar, Havdin, Hadrure
| Suray, Bali, Guhan, Sele, Bezih, Zaza, Zacet, Saya, Dasen, Sevdam | Kurt, Iyarlan, Kagiy, Kiysan, Samay, Bervan, Safi, Fur, Zetir, Biheti
| Beyfa, Fahat, Dula, Zatkun, Semelten, Gondur, Gulerun, Buysut, Gukpek, Alpfehatengel | Gocdan, Buhac, Sabip, Belet, Iben, Buhise, Aran, Semum, Sencin, Ceni 
| Aseye, Emra, Fera, Oyas, Zumul, Azel, Fular, Guli, Saner, Fasi | Urman, Erikmar, Buhan, Rahif, Hadul, Rosen, Daban, Hana, Rani, Zizasel 
| Cevli, Seylut, Sayih, Gece, Kaki, Dilsin, Semi, Aser, Sabil, Boylu | Heri, Soket, Serme, Selzikcan, Nurlay, Harfir, Seyyulga, Zeyar, Dilal, Gunsan



## My thoughts
For me, the results when the model is trained with the 10K dataset sound more familiar. Probably that's because I only heard a certain amount of Turkish names in my lifetime and when looking at the whole dataset, there are many many names which feel weird / funny to me :) Naturally, these types of weird sounding Turkish names are more present in the 3m dataset, so what my model generates may sound more like them. 

Also what made me happy is that all the generated names are at least pronouncible in Turkish. Normally in Turkish, no more than 2 consonants appear next to each other. A vowel is usually followed by a consonant and a consonant is usually followed by another consonant or a vowel. The model seems to have followed this rule with a high percentage of the names it generated so I'm at least interpreting this as a good result. 

And the most fun part of this doing this project was to loudly laugh at the funny names in the dataset or the names my model predicted. So at the very least, having a good laugh was worth the time I spent on this :]





