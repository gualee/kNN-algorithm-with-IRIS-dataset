# kNN-algorithm-with-IRIS-dataset
### 透過Python撰寫kNN分類演算法，來分類鳶尾花數據集

### 以下為演算法步驟：
1.	載入wine資料集，做正規化
2.	利用歐式距離計算測試點與每個點的距離
3.	排序前五個最短的距離
4.	判斷這五個點大部分屬於哪一個類別，將之判定為此類別
5.	計算準確率以及執行時間

一般在實作knn演算法時，sklearn套件裡面有幾種主要演算法去實作，有暴力法(brute-force)，KD樹(KDTree)和球樹(BallTree)，其它的有BBF樹和MVP樹。而計算距離(distance metric)的方式亦有好幾種，有歐式距離(Euclidean distance)、曼哈頓距離(Manhattan distance)、Cosine Similarity、Mahalanobis distance、Chebyshev Distance這裡選擇歐式距離來測量測試點和其他點的距離。數值正規化的方式亦有好幾種，min-max normalization、Z-score normalization、log函數轉換及atan函數轉換等，這裡使用第一種，將數據縮放到[0,1]的區間，避免某些特徵的值太大而影響到整體的訓練結果。不過此種有其缺點，就是新數據加入時，會導致min及max需重新計算，在後續做法可以使用經驗常量來取代min和max值。K值的部分在sklearn套件內預設是5，而我們使用wine資料集因為有3種類別，所以選擇k為5為一個合理的選擇。執行時間的計算方式為從for迴圈開始進入knn函式前開始計時，到迴圈結束時停止計時，分別計算訓練時間及測試時間。準確率計算方式為將分類結果與分類標籤作比對，將錯誤的累計做加總，最後再用1扣掉錯誤的部分，分母為資料集數量(訓練集和測試集各50%)，最後以百分比方式呈現準確率。
