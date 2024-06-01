# mie-pathlogy

## ToDo:
- TCGA pre-training
- Unsupervised learning, higher order embedding


## Dataset
- [The Cancer Genome Atlas Program - National Cancer Institute](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)
    - [GDC Data Portal](https://gdc.cancer.gov/access-data/gdc-data-portal)
        - [Repository - GDC Data Portal](https://portal.gdc.cancer.gov/repository?facetTab=files&filters={"op"%3A"and"%2C"content"%3A[{"op"%3A"in"%2C"content"%3A{"field"%3A"files.data_type"%2C"value"%3A["Slide Image"]}}]})あたり?

### Private dataset
- Header:
    - use:
        - 0: 使用しない
        - 1: 学習データ
        - 2: 評価データ
    - number: 被験者番号
    - 転帰: とりあえず今は気にしない
    - survival
        - 0:生存、1:死亡
    - survival time: 単位は月
    - event:
        - naは評価不能または最初からイベントが発生済み
    - event time

#### ファイル
- `survival.csv`: 生存・非生存
    - `survival_3os.csv`: 3年時生存
    - `survival_2dfs.csv`: 2年時無病生存


#### リンク
`/net/nfs2/export/dataset/mie-ortho/pathology`内に保存しているので、
```shell-session
$ cd ~/workspace/mie-pathology/_data
$ ln -s /net/nfs2/export/dataset/mie-ortho/pathology/svs
$ ln -s /net/nfs2/export/dataset/mie-ortho/pathology/xml
```
でローカルリポジトリ内にシンボリックリンクを貼って使う。

## Code

```shell-session
$ sudo apt install -y libopenslide-dev
```

```shell-session
$ pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Refactoring (survival_time)
**コードは一部ロールバックしてるかも… 論文の方が正確**

```shell
$ cd survival_time
$ PYTHONPATH=. python3 baseAE.py
$ PYTHONPATH=. python3 baseCL.py
$ PYTHONPATH=. python3 train_time.py
```

- [x] データは今セットしてあるCSVでOK
- [x] Base model
    - [x] AutoEncoder.py -> baseAE.py 動かす：3-4days
        - [x] だけ
    - [x] contrastive_learning.py -> baseCL.py 動かす:  2days /100epoch ぐらい
        - [x] SupCon（Loss）はうまくいかなかったので使ってない
        - [x] NCE+CEだけでやったから、Lossは変更不要
- [ ] Class分類モデル（train_time.py　動かす）：各2days
    - [ ] 学習のループないで、Hard/Softの切り替え（コメント１行入れ替えるだけ）、
    - [ ] +ロスの切り替えはCE,SoftはKL-Divにコメントで切り替え（学習ループ前）、それぞれ関数あり
    - [ ] PatchDatasetでSoftの種類を切り替え
    - [ ] AutoEncode/CLの切り替え：L399ぐらいからがAE、L419-454ぐらいまでがCL
    - [ ] ResNetでシンプルくらす分類：L455ぐらいから-L467ぐらいまで
        - [ ] ResNetの時はPatchDtaasetのTransofrm,で、Reisze, Normalizeも有効化する
    - [ ] Transformer
        - [ ] L471ぐらいから
        - [ ] CLはやってないから、無視でも良い
        - [ ] AE: L482-
        - [ ] PatchDataset.__init__()
            - [ ] L60ぐらいから→ CNN
            - [ ] L67ぐらいからL70 →　Transformer
        - [ ] 画像まとめるやす（PatchDataset.__Getitem__）
            - [ ] L145: CNN
            - [ ] L150: Transformer
            - [ ] Returnも切り替える！！
    - [ ] 評価
        - [ ] graph-plot.py: Logのパス変える
            - [ ] AE, CLそのもの、train_time…
        - [ ] 正解値 vs. 推定値のプロット（plot_predvstrue.py）
            - [ ] train_timeで学習したモデルのpthのパスを切り替える。 L229前後
            - [ ] その前のモデルの定義も変える
            - [ ] **load_state_dictからload_modelに帰れるようにする？？？**
        - [ ] plot_hist.py
            - [ ] 推定生存期間のHistogramを作る
            - [ ] Modelと学習済みパラメータの切り替えがひつよう。
