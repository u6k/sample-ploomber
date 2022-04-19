# sample-ploomber

Ploomberを素振りする

## Usage

ビルドする。

```bash
docker-compose build
```

Pipenvで実行できるスクリプトを表示する。

```bash
docker-compose run app
```

訓練パイプライン処理を実行する。

```bash
docker-compose run app pipenv run pipeline
```

推論パイプライン処理を実行する。

```bash
docker-compose run app pipenv run pipeline --entry-point pipeline-predict.yaml
```
