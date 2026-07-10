# Fish Speech 1.5 ローカル生成環境 (M2 Max向け設定)

Fish Speech 1.5を使って、テキストから音声を生成するための環境です。
M2 Maxチップ搭載Macでの安定動作（CPUモード実行）に最適化されています。

## 🚀 すぐに使う

### 1. 仮想環境の有効化
コマンドを実行する前に、必ずこの仮想環境をアクティベートしてください。

```bash
cd /Users/taku.me/Library/CloudStorage/Dropbox/_inbox/AppData/FishSpeech
source .venv/bin/activate
```

### 2. 音声生成を実行
`generate.py` スクリプトを使って、好きなテキストを喋らせることができます。

**基本の使い方:**
```bash
python generate.py "こんにちは、架空のラジオへようこそ！"
```
これで `output.wav` というファイルが生成されます（デフォルト音声： `opening.mp3`）。

---

## 🛠 詳細な使い方

### ファイル名を指定して保存
```bash
python generate.py "今日の天気は晴れです" -o weather.wav
```

### 別の声を使う (ボイスクローン)
別の音声ファイルを指定することで、その声を真似させることができます。
```bash
python generate.py "次のニュースです" --ref_audio "/path/to/another_voice.mp3"
```

### クオリティを上げるコツ
参照音声（Reference Audio）が実際に喋っている内容をテキストで与えることで、イントネーションの再現度が向上します。
```bash
python generate.py "こんにちは" --ref_audio "voice.mp3" --ref_text "voice.mp3の文字起こしテキスト"
```

---

## ⚠️ 注意点
- **実行デバイス:** デフォルトで `cpu` モードで動作します。
  - これは Fish Speech の大規模モデルを Mac の MPS (GPU) で動かした際に発生する `NotImplementedError` を回避するためです。
- **生成時間:** 短い文章であれば、M2 Max CPUで10秒〜20秒程度で生成されます。
- **デフォルト音声:** AI News Podcastで使用されている声（Voice ID: `5161d41404314212af1254556477c17d`）がデフォルトで適用されます。
  - 参照元: `/Users/taku.me/.../ai-news-podcast/remotion-project/public/audio/opening.mp3`
  - 最適化: 参照音声の正確な書き起こしテキストを自動的に使用するため、声質と抑揚の再現度が向上しています。

## 📁 ファイル構成
- `generate.py`: 音声生成用のメインスクリプト
- `README.md`: この説明書
- `.venv/`: 必要なライブラリが全て入った仮想環境
- `checkpoints/`: Fish Speech 1.5 と Firefly GAN のモデルファイル
