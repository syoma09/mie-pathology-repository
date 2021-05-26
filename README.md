# mie-pathlogy

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
        - 1: 生存、0:死亡
    - survival time: 単位は月
    - event:
        - naは評価不能または最初からイベントが発生済み
    - event time

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


