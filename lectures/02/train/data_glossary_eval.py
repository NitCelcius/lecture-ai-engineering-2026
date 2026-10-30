# 第2回 宿題(発展): 用語集 SFT の評価質問。
# 学習データ(用語集 58 語 x 2 テンプレート)と質問文は重ならない。
# kind: reworded = 用語集にある語を言い回しを変えて聞く / alias = 別名で聞く / out = 用語集にない語
# IN_GLOSSARY: (term id, 質問, 答えに含まれるべき語[すべて])。OUT_OF_GLOSSARY: 質問のみ。

IN_GLOSSARY = [
    ("pretraining", "事前学習では何をするのですか？", "reworded", ["次のトークン"]),
    ("sft", "SFT ではどんなデータを使いますか？", "reworded", ["ペア"]),
    ("lora", "LoRA では何を学習するのですか？", "reworded", ["小さな行列"]),
    ("qlora", "QLoRA はどうやってメモリを節約するのですか？", "reworded", ["4ビット"]),
    ("overfitting", "過学習にはどうやって気づきますか？", "reworded", ["検証データ"]),
    ("epoch", "エポックを増やしすぎるとどうなりますか？", "reworded", ["過学習"]),
    ("learning-rate", "学習率を大きくしすぎるとどうなりますか？", "reworded", ["不安定"]),
    ("huggingface-hub", "Hugging Face Hub では何ができますか？", "reworded", ["アダプタ"]),
    ("api-key", "この講座では API キーをどこに登録しますか？", "reworded", ["Codespaces secret"]),
    ("default-model", "この講座の演習で標準に使うモデルは何ですか？", "reworded", ["gemini-3.5-flash-lite"]),
    ("human-in-the-loop", "Pydantic AI で人の承認を挟むにはどうしますか？", "reworded", ["requires_approval"]),
    ("structured-output", "Pydantic AI で構造化出力を受け取るには何を指定しますか？", "reworded", ["output_type"]),
    ("test-model", "TestModel は何のために使いますか？", "reworded", ["ダミー"]),
    ("pi", "Pi の npm パッケージ名を教えてください。", "reworded", ["@earendil-works/pi-coding-agent"]),
    ("harness", "Pi の組み込みツールは何ですか？", "reworded", ["read", "write", "edit", "bash"]),
    ("adk", "ADK の 2.0 では何が変わりましたか？", "reworded", ["破壊的変更"]),
    ("mcp-transport", "stdio と Streamable HTTP はどう違いますか？", "reworded", ["サブプロセス", "URL"]),
    ("mcp-config", "mcpServers 設定にはサーバーごとに何を書きますか？", "reworded", ["起動コマンド", "URL"]),
    ("tool-namespace", "複数の MCP サーバーのツール名が衝突しないようにするには？", "reworded", ["接頭辞"]),
    ("fetch-server", "fetch サーバーはどのコマンドで起動しますか？", "reworded", ["uvx mcp-server-fetch"]),
    ("homework-levels", "宿題にはどんなレベルがありますか？", "reworded", ["基礎", "実装", "発展"]),
    ("lec-start-tag", "lec03-start のようなタグは何のためにありますか？", "reworded", ["ブランチ"]),
    ("codespaces", "Codespaces を使わないときはどうするべきですか？", "reworded", ["停止"]),
    ("github-copilot", "Copilot Free にはどんな制限がありますか？", "reworded", ["補完の回数"]),
    ("hallucination", "ハルシネーションを抑えるにはどうしますか？", "reworded", ["根拠"]),
    ("rate-limit", "レート制限を超えるとどんなエラーが返りますか？", "reworded", ["429"]),
    ("temperature", "temperature を低くするとどうなりますか？", "reworded", ["安定"]),
    ("guardrail", "ガードレールにはどんな方法がありますか？", "reworded", ["人の確認"]),
    ("lora", "Low-Rank Adaptation とは何ですか？", "alias", ["行列"]),
    ("fine-tuning", "追加学習とは何ですか？", "alias", ["事前学習済み"]),
    ("human-in-the-loop", "HITL とは何ですか？", "alias", ["承認"]),
    ("rate-limit", "429エラーが出るのはなぜですか？", "alias", ["上限"]),
    ("tool-description", "docstring はツールでどんな役割をしますか？", "alias", ["自動"]),
    ("mcp-config", "mcp.json には何を書きますか？", "alias", ["MCPサーバー"]),
    ("coding-agent", "Coding Agent の中身はどんな構造ですか？", "alias", ["ループ"]),
    ("peft", "Parameter-Efficient Fine-Tuning とは何ですか？", "alias", ["一部"]),
]

OUT_OF_GLOSSARY = [
    "DPO とは何ですか？",
    "RLHF とは何ですか？",
    "蒸留とは何ですか？",
    "Transformer とは何ですか？",
    "プロンプトインジェクションとは何ですか？",
    "ベクトルデータベースとは何ですか？",
    "pytest とは何ですか？",
    "Mixture of Experts とは何ですか？",
    "AIE2026 の第4回の宿題の締切はいつですか？",
    "AIE2026 の講師は誰ですか？",
]

ABSTAIN_MARKERS = ["記載がありません", "記載されていません", "載っていません", "わかりません", "分かりません", "情報がありません", "確認が必要", "見つかりません"]
