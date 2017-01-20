執行順序為

python Compound.py data/test.csv  ---->  compound.txt
此程式是用來找出 Compound word ( Bigram/Trigram )。

python CleanText.py data/test.csv ---->  cleantext.txt
此程式是用來清潔並加入 Bigram word 用來使用 Word-to-Vector。

python W2V.py ----> w2vmodel
產生 Word-to-Vector Model。

python w2v_tfidf.py data/test.csv ---->  sub_tfidf_final.csv
產生最後的預測結果。