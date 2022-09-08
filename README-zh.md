# 華語-臺語機械翻譯

本專案使用[臺灣閩南語常用詞詞典](https://twblg.dict.edu.tw/holodict_new/)作為訓練語料，基於中文 BART 模型 [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) 微調，實現了華語-台語機械翻譯。

## 任務描述

本專案解決的任務是華語-臺語機械翻譯，模型的輸入為華語句子，預期的輸出為對應的臺語句子，例如：

```
輸入：你的文章寫得很好，可以去投稿了。
輸出：你的文章寫了真好，會使去投稿矣。
```

## 環境配置

本專案中的腳本適合於單機單卡的 GPU 機器使用。

```sh
python3.10 -m venv ./venv
. ./venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[cuda]==0.3.17" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

## 資料集

本專案使用的資料集為[臺灣閩南語常用詞詞典](https://twblg.dict.edu.tw/holodict_new/)，資料蒐集於 [g0v/moedict-data-twblg](https://github.com/g0v/moedict-data-twblg/blob/master/uni/%E4%BE%8B%E5%8F%A5.csv)。本專案選用這一資料，是因為資料集中的例句源自詞典，而詞典中往往收錄更多臺灣閩南語特有字詞，因此例句所用詞彙更為豐富，可以使模型學習到更加完整的臺灣閩南語用法。

資料集存放在 `lib/twblg/data.tsv`，其中包含 8366 對句子，每對句子包含華語及對應的臺語翻譯。將句子亂數重排後，依比例 8:1:1 將資料集分割為訓練集、開發集和測試集。

TODO: 目前未有使用開發集

## 資料預處理

本專案基於 [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) 模型，而該模型使用簡體中文，因此輸入模型的字串及模型輸出也應為簡體中文。

為此，最一般的做法是使用簡繁轉換程式將華語及臺語句子轉換為簡體中文，訓練模型在輸入簡體華語句子時，輸出為簡體的臺語句子。當使用模型時，同樣將輸入的句子轉換為簡體輸入模型，待模型輸出簡體的臺語句子後，再使用簡繁轉換程式轉換為繁體。但是，這樣做存在兩方面的問題：

第一，對於已知繁體原文的句子，由於「一簡對多繁」現象的存在，將句子轉換為簡體中文後，會造成訊息的損失。之後若想再次使用繁體原文，如果直接將簡體中文文本轉換回繁體，即使在考慮上下文內容的情況下，亦不能保證轉換完全準確。例如，簡體中文句子「他偷偷给我发卡」中，「发卡」可能對應繁體中文的「發卡」（簽發卡片）或「髮卡」（髮夾）。

第二，對於模型生成的新的簡體句子，使用簡繁轉換程式轉換為繁體。雖然對於華語文本，目前內置轉換詞庫的簡繁轉換程式已經可以保證較高的準確性，但對於臺語文本而言，由於詞庫中缺少臺語特有詞彙，轉換的準確性較低。例如，將簡體詞語「代志」轉換為繁體時，由於詞庫中缺少該詞，會將詞語拆開分別轉換，而「志」字可能對應繁體中文的「志」與「誌」，程式可能優先選擇「志」字，從而導致轉換錯誤。

為了解決上述兩個問題，我在使用常見的簡繁轉換程式 [StarCC](https://github.com/StarCC0/starcc-py) 的基礎上，對方法進行了兩處改進：

第一，將繁體句子轉換為簡體時保留繁體原文，以供未來使用。

第二，將繁體句子轉換為簡體時，根據臺語句子中已經給定的斷詞訊息（臺羅拼音本身是以詞劃分）將臺語詞彙存入轉換表中。在將新句子由簡體轉換為繁體時，使用新的轉換表。

## 斷詞器

本專案基於 [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) 模型，而該模型使用的斷詞器為目前使用範圍甚廣的為詞表長度 21128 的中文 BERT 斷詞器。然而，該斷詞器存在四個方面的問題：

第一，BERT 斷詞器使用 WordPiece 斷詞演算法。對於不含英文單字的純中文文本，直接使用基於單字的斷詞演算法已經足夠，而不需要使用 WordPiece。使用 WordPiece 帶來的後果是，詞表中有 7322 個以 `##` 起始的詞，表示詞的中間部分，例如 `##力`。而由於中文以字為單位的特性，這些詞在實際斷詞中根本不會出現，造成了大量的資源浪費。

第二，儘管中文 BART 模型是使用簡體語料訓練，詞表中仍然存在大量繁體中文漢字。可以確定的是，由於預訓練語料為簡體，即使預訓練語料中存在少量繁體字，也不足以令模型學到有意義的內容。即使模型在一定程度上學習到這些繁體字的意義，由於本專案中輸入的資料經過簡繁轉換，這些繁體字在實際斷詞中根本不會出現，造成了大量的資源浪費。

第三，詞表中存在日文、韓文等字符和大量特殊符號，這在本專案中不會出現，造成了大量的資源浪費。

第四，詞表是為華語文本設計，沒有包含臺語特有的字，例如缺少常用的「𪜶」字。如果詞表缺少這些字，程式在繙譯到需要使用這些字的句子時就會出現錯誤。

為了解決上述四個問題，我分別採用如下方法：

第一，由於我準備使用基於單字的斷詞演算法而非 WordPiece 斷詞演算法，我將 7322 個以 `##` 起始的詞全部移除。

第二，（TODO: 大坑）

第三，我通過人工判斷將日文、韓文等字符和無用的特殊符號移除。

第四，我將所有在訓練語料中出現，而詞表中沒有的字加入詞表中。

由此得到長度為 7697 的新詞表。

TODO: VSCode 仍然抱怨 unusual line terminators

## Embedding

Removed

Randomly initialised.

## Dataloader

Since the dataset is very small, the data are directly placed on the GPU memory. However, for larger dataset, the `main` branch provides an on-demand dataloader.

## Training

SGD

## Generation

## Analysis

對於斷詞器，未知的單字不應該直接給 UNK，而是應該 COPY。
